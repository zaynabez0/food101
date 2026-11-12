"""Feature caching (Phase 4): run the trained image and text models once over every split.

For each split, saves to <features_dir>/<split>.pt a dict of row-aligned tensors (row i = same sample):
    image_features (N, 2048)  image_probs (N, 101)  text_features (N, 128)  text_probs (N, 101)
    labels (N,)  sample_ids [N image filenames]  checkpoints {image/text: {path, epoch, val_acc}}
No augmentation on any split (eval transforms, also for train) and no shuffling.
"""
import os
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data import FoodDataset, load_tokenizer
from src.train import get_device


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    """Load best.pt or last.pt weights into model. Out: {path, epoch, val_acc} of those weights."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    # best.pt stores val_acc directly; last.pt stores the history, whose last row matches its weights
    val_acc = ckpt["val_acc"] if "val_acc" in ckpt else ckpt["history"][-1]["val_acc"]
    print(f"loaded {path} (epoch {ckpt['epoch']}, logged val acc {val_acc:.4f})")
    # the epoch matters: last.pt keeps the same path while training moves on
    return {"path": path, "epoch": ckpt["epoch"], "val_acc": val_acc}


@torch.inference_mode()
def extract_split(
    split: str,
    image_model: nn.Module,
    text_model: nn.Module,
    config: dict,
    device: torch.device,
) -> dict:
    """Run both models over one split in file order. Out: dict of row-aligned CPU tensors.
    config["max_items"] (optional, smoke runs) keeps only the first N rows."""
    dataset = FoodDataset(split, "both", config, load_tokenizer(config), augment=False)
    n = min(config.get("max_items") or len(dataset), len(dataset))
    loader = DataLoader(Subset(dataset, range(n)), batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])
    use_amp = config["amp"] and device.type == "cuda"
    parts = {k: [] for k in ("image_features", "image_probs", "text_features", "text_probs", "labels")}

    for image, input_ids, attention_mask, labels in loader:
        # One loader in "both" mode: image[i] and input_ids[i] come from the same CSV row.
        image, input_ids, attention_mask = image.to(device), input_ids.to(device), attention_mask.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            img_feat = image_model.extract_features(image)                     # (B, 2048)
            img_logits = image_model.classify(img_feat)                       # (B, 101)
            txt_feat = text_model.extract_features(input_ids, attention_mask)  # (B, 128)
            txt_logits = text_model.classify(txt_feat)                        # (B, 101)
        parts["image_features"].append(img_feat.float().cpu())
        parts["image_probs"].append(img_logits.float().softmax(dim=1).cpu())
        parts["text_features"].append(txt_feat.float().cpu())
        parts["text_probs"].append(txt_logits.float().softmax(dim=1).cpu())
        parts["labels"].append(labels)

    out = {k: torch.cat(v) for k, v in parts.items()}
    out["sample_ids"] = dataset.sample_ids[:n]
    check_alignment(out, dataset.labels[:n])
    return out


def check_alignment(out: dict, expected_labels: torch.Tensor) -> None:
    """Raise if the saved rows are not exactly the first len(expected_labels) dataset rows, in order."""
    n = len(expected_labels)
    for key in ("image_features", "image_probs", "text_features", "text_probs", "labels"):
        if out[key].shape[0] != n:
            raise RuntimeError(f"{key} has {out[key].shape[0]} rows, dataset has {n}")
    if not torch.equal(out["labels"], expected_labels):
        raise RuntimeError("labels collected from the loader are not in dataset order")
    if len(out["sample_ids"]) != n or len(set(out["sample_ids"])) != n:
        raise RuntimeError("sample_ids missing or not unique")
    print(f"  alignment ok: {n} rows, image/text/labels/sample_ids in the same order")


def accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy of (N, C) probabilities against (N,) labels."""
    return (probs.argmax(dim=1) == labels).float().mean().item()


def run_extraction(
    config: dict,
    image_model: nn.Module,
    text_model: nn.Module,
    on_split_done: Callable[[], None] | None = None,
) -> None:
    """Load checkpoints, extract every split, save it, and print the accuracy sanity check."""
    device = get_device()
    image_model, text_model = image_model.to(device).eval(), text_model.to(device).eval()
    logged = {
        "image": load_checkpoint(image_model, config["image_checkpoint"], device),
        "text": load_checkpoint(text_model, config["text_checkpoint"], device),
    }
    os.makedirs(config["features_dir"], exist_ok=True)

    for split in config["splits"]:
        print(f"\n[{split}]")
        out = extract_split(split, image_model, text_model, config, device)
        out["checkpoints"] = logged   # {image: {path, epoch, val_acc}, text: {...}}
        path = os.path.join(config["features_dir"], f"{split}.pt")
        torch.save(out, path)
        print(f"  saved {path}: " + ", ".join(
            f"{k} {tuple(v.shape)}" for k, v in out.items() if isinstance(v, torch.Tensor)))

        if split in ("val", "test"):
            for name in ("image", "text"):
                acc = accuracy(out[f"{name}_probs"], out["labels"])
                note = f" (training log val acc {logged[name]['val_acc']:.4f})" if split == "val" else ""
                print(f"  {name:>5} acc from saved softmax: {acc:.4f}{note}")
        if on_split_done is not None:
            on_split_done()
