"""Train a fusion model on the cached features of src/extract.py with the generic loop of src/train.py.

Late fusion (Table III): inputs = image_probs (N, 101) + text_probs (N, 101) of each split.
Early fusion (Table IV): inputs = image_features (N, 2048) + text_features (N, 128).
- train split: gradient updates; val split: best epoch, LR drops, early stopping (inside fit);
- test split: evaluated once, at the end, with the best-on-val weights.
Results: /ckpt/results/{late_<merge>,early}.json = per-epoch history + test metrics (+ learned alpha);
early fusion also writes /ckpt/results/early_test_predictions.csv (one row per test sample).
"""
import csv
import json
import os
import shutil
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion import build_early_fusion, build_late_fusion
from src.train import build_optimizer, build_scheduler, evaluate, fit, get_device, set_seed

SPLITS = ("train", "val", "test")
LATE_INPUTS = ("image_probs", "text_probs")         # keys of {split}.pt, in the model's argument order
EARLY_INPUTS = ("image_features", "text_features")  # (N, 2048), (N, 128)


def load_features(features_dir: str, splits: tuple[str, ...] = SPLITS) -> dict[str, dict]:
    """Load <features_dir>/{split}.pt for each split. Out: {split: dict saved by src/extract.py}."""
    features = {}
    for split in splits:
        path = os.path.join(features_dir, f"{split}.pt")
        features[split] = torch.load(path, map_location="cpu", weights_only=True)
        print(f"loaded {path}: {features[split]['labels'].shape[0]} rows")
    return features


def check_same_checkpoints(features: dict[str, dict], splits: tuple[str, ...] = SPLITS) -> dict:
    """Warn if the splits were extracted from different model checkpoints. Out: the first split's metadata."""
    first = splits[0]
    reference = features[first].get("checkpoints")
    for split in splits[1:]:
        if features[split].get("checkpoints") != reference:
            print(f"WARNING: {split}.pt was extracted from other checkpoints than {first}.pt:\n"
                  f"  {first}: {reference}\n  {split}: {features[split].get('checkpoints')}\n"
                  "  -> re-run extract_features on all splits before reporting numbers.")
    return reference


def make_loader(split_features: dict, input_keys: tuple[str, ...], batch_size: int, shuffle: bool) -> DataLoader:
    """TensorDataset(*inputs, labels) from one split's cached tensors. Batch: [(B, 101), (B, 101), (B,)]."""
    tensors = [split_features[key] for key in input_keys] + [split_features["labels"]]
    # Everything already sits in memory as tensors: no worker processes needed.
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy of (N, C) scores against (N,) labels."""
    return (probs.argmax(dim=1) == labels).float().mean().item()


def baseline_accuracy(split_features: dict) -> dict[str, float]:
    """No-training baselines the fusion models must beat. Out: {image, text, average} accuracy.

    average = (p_img + p_txt) / 2: each class gets the mean of the two models' probabilities.
    """
    p_img, p_txt = split_features["image_probs"], split_features["text_probs"]   # (N, 101) each
    labels = split_features["labels"]                                            # (N,)
    return {
        "image": accuracy(p_img, labels),
        "text": accuracy(p_txt, labels),
        "average": accuracy((p_img + p_txt) / 2, labels),                        # (N, 101)
    }


def evaluate_baselines(features_dir: str, results_path: str, splits: tuple[str, ...] = ("val", "test")) -> dict:
    """Baseline accuracies on val (for comparison) and test (reported); saved as JSON. Out: {split: {...}}."""
    features = load_features(features_dir, splits)
    results = {
        "feature_checkpoints": check_same_checkpoints(features, splits),
        **{split: baseline_accuracy(features[split]) for split in splits},
    }
    write_json(results_path, results)
    for split in splits:
        print(f"{split:>5}: " + " | ".join(f"{k} {v:.4f}" for k, v in results[split].items()))
    return results


def train_fusion(
    config: dict,
    build_model: Callable[[dict], nn.Module],
    input_keys: tuple[str, ...],
    run_name: str,
    ckpt_root: str,
    on_epoch_end: Callable[[], None] | None = None,
    save_test_predictions: bool = False,
) -> dict:
    """Train on train, select on val (fit), test once. Out: the final results dict (also saved as JSON).

    save_test_predictions: also write results/<run_name>_test_predictions.csv (see write_test_predictions).
    """
    set_seed(config["seed"])                       # before building the model: same init every run
    device = get_device()
    features = load_features(config["features_dir"])
    feature_checkpoints = check_same_checkpoints(features)

    ckpt_dir = os.path.join(ckpt_root, run_name)
    results_path = os.path.join(ckpt_root, "results", f"{run_name}.json")
    # CHOICE: a fusion run takes minutes, so it always starts fresh instead of resuming an old
    # last.pt (which could come from another config or other features).
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    model = build_model(config).to(device)
    train_loader = make_loader(features["train"], input_keys, config["batch_size"], shuffle=True)
    val_loader = make_loader(features["val"], input_keys, config["batch_size"], shuffle=False)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    history = fit(model, train_loader, val_loader, optimizer, scheduler, config, "fusion",
                  ckpt_dir, results_path, on_epoch_end)

    # --- the only use of the test split: best-on-val weights, evaluated once ---
    best = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    test_loader = make_loader(features["test"], input_keys, config["batch_size"], shuffle=False)
    test_metrics = evaluate(model, test_loader, device, "fusion", use_amp=False)
    if save_test_predictions:
        predictions_path = os.path.join(ckpt_root, "results", f"{run_name}_test_predictions.csv")
        write_test_predictions(predictions_path, features["test"], predict(model, test_loader, device))

    results = {
        "config": config,
        "feature_checkpoints": feature_checkpoints,        # which image/text models made the inputs
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "baseline_test_acc": baseline_accuracy(features["test"]),   # image / text / average
        "history": history,
    }
    alpha = getattr(model, "alpha", None)                   # LateFusion, weighted mode only
    if alpha is not None:
        results["alpha"] = alpha.item()                     # weight on the image softmax
    write_json(results_path, results)
    print_summary(run_name, results)
    return results


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Predicted class of every row, in loader order (shuffle=False). Out: (N,) int64 on CPU."""
    model.eval()
    preds = []
    for *inputs, _ in loader:                                   # batch = (*inputs, labels)
        logits = model(*(t.to(device) for t in inputs))         # (B, 101)
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds)


def write_test_predictions(path: str, test_features: dict, fusion_preds: torch.Tensor) -> None:
    """CSV, one row per test sample: sample_id, label, image_pred, text_pred, fusion_pred (class indices)."""
    if fusion_preds.shape[0] != test_features["labels"].shape[0]:
        raise RuntimeError("fusion predictions and test features do not have the same number of rows")
    columns = {
        "sample_id": test_features["sample_ids"],                        # image filename
        "label": test_features["labels"].tolist(),
        "image_pred": test_features["image_probs"].argmax(dim=1).tolist(),
        "text_pred": test_features["text_probs"].argmax(dim=1).tolist(),
        "fusion_pred": fusion_preds.tolist(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)                                         # header
        writer.writerows(zip(*columns.values()))                         # row i = sample i
    print(f"saved {path}: {len(columns['label'])} rows")


def write_json(path: str, data: dict) -> None:
    """Write data as indented JSON (overwrites the per-epoch file written by fit)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_summary(run_name: str, results: dict) -> None:
    """One block with the numbers that matter for the report."""
    print(f"\n[{run_name}] best epoch {results['best_epoch']} | val acc {results['best_val_acc']:.4f}"
          f" | test acc {results['test_acc']:.4f}")
    base = results["baseline_test_acc"]
    print(f"  baselines on test: image {base['image']:.4f} | text {base['text']:.4f}"
          f" | average {base['average']:.4f}")
    if "alpha" in results:
        print(f"  learned alpha (image weight) {results['alpha']:.4f}, text weight {1 - results['alpha']:.4f}")


def train_late_fusion(
    config: dict, ckpt_root: str, on_epoch_end: Callable[[], None] | None = None
) -> dict:
    """Late fusion (Table III) on the cached softmax vectors; run name late_{merge}."""
    return train_fusion(config, build_late_fusion, LATE_INPUTS, f"late_{config['merge']}",
                        ckpt_root, on_epoch_end)


def train_early_fusion(
    config: dict, ckpt_root: str, on_epoch_end: Callable[[], None] | None = None
) -> dict:
    """Early fusion (Table IV) on the cached image + text features; run name "early", with test CSV."""
    return train_fusion(config, build_early_fusion, EARLY_INPUTS, "early",
                        ckpt_root, on_epoch_end, save_test_predictions=True)
