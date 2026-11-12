"""Inference on new (image, title) pairs with the best fusion model, chosen on validation accuracy.

In:  raw image files (bytes) + titles (str), the trained checkpoints in <ckpt_root>
Out: for each pair, the top-k classes (with probabilities) of the image model, the text model and the fusion.
"""
import glob
import io
import json
import os

import pandas as pd
import torch
from PIL import Image

from src.data import build_image_transform, clean_title, load_tokenizer, tokenize_titles
from src.extract import load_checkpoint
from src.models.fusion import build_early_fusion, build_late_fusion
from src.models.image import build_image_model
from src.models.text import build_text_model
from src.train import get_device


def list_fusion_runs(results_dir: str) -> dict[str, dict]:
    """All fusion results JSON (late_*, early, early_seed*). Out: {run_name: results dict}."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "late_*.json"))
                       + glob.glob(os.path.join(results_dir, "early*.json"))):
        runs[os.path.splitext(os.path.basename(path))[0]] = json.load(open(path))
    if not runs:
        raise FileNotFoundError(f"no fusion results in {results_dir}: train a fusion model first")
    return runs


def select_best_fusion(runs: dict[str, dict]) -> str:
    """Run name with the highest best_val_acc. The test set plays no part in this choice."""
    for name, run in sorted(runs.items(), key=lambda kv: -kv[1]["best_val_acc"]):
        print(f"  {name:<16} val acc {run['best_val_acc']:.4f}")
    return max(runs, key=lambda name: runs[name]["best_val_acc"])


class FusionPredictor:
    """Image model + text model + fusion head, loaded once; predict() runs all three on new pairs."""

    def __init__(
        self,
        image_cfg: dict,
        text_cfg: dict,
        ckpt_root: str,
        class_names: list[str],
        run_name: str | None = None,
    ) -> None:
        """run_name: fusion run to use ("early", "late_sum", ...); None = best on validation."""
        self.device = get_device()
        self.class_names = class_names
        self.text_cfg = text_cfg
        self.transform = build_image_transform(image_cfg, train=False)   # same as val/test
        self.tokenizer = load_tokenizer(text_cfg)

        runs = list_fusion_runs(os.path.join(ckpt_root, "results"))
        print("fusion runs (val accuracy):")
        best = select_best_fusion(runs)
        self.run_name = run_name or best
        if self.run_name not in runs:
            raise ValueError(f"unknown fusion run {self.run_name!r}, available: {sorted(runs)}")
        run = runs[self.run_name]
        self.val_acc, self.test_acc = run["best_val_acc"], run["test_acc"]
        self.is_late = self.run_name.startswith("late_")
        print(f"-> using '{self.run_name}' (val {self.val_acc:.4f}, test {self.test_acc:.4f})")

        # Base models: the exact checkpoints that produced the features the fusion head was trained on.
        # pretrained=False: no ImageNet download, the checkpoint overwrites every weight anyway.
        self.image_model = build_image_model({**image_cfg, "pretrained": False}).to(self.device).eval()
        self.text_model = build_text_model(text_cfg).to(self.device).eval()
        load_checkpoint(self.image_model, run["feature_checkpoints"]["image"]["path"], self.device)
        load_checkpoint(self.text_model, run["feature_checkpoints"]["text"]["path"], self.device)

        # Fusion head: rebuilt from the config saved in its results JSON (same merge mode, sizes).
        build = build_late_fusion if self.is_late else build_early_fusion
        self.fusion_model = build(run["config"]).to(self.device).eval()
        load_checkpoint(self.fusion_model, os.path.join(ckpt_root, self.run_name, "best.pt"), self.device)

    @torch.inference_mode()
    def predict(self, images: list[bytes], titles: list[str], top_k: int) -> list[dict]:
        """Raw image files + titles (same length) -> one dict per pair with top-k of each model."""
        if len(images) != len(titles):
            raise ValueError(f"{len(images)} images but {len(titles)} titles")
        pixels = torch.stack([self.transform(Image.open(io.BytesIO(b)).convert("RGB"))
                              for b in images]).to(self.device)                  # (B, 3, 299, 299)
        clean = [clean_title(t) for t in titles]
        input_ids, attention_mask = tokenize_titles(clean, self.tokenizer, self.text_cfg["max_length"])
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)  # (B, L)

        img_feat = self.image_model.extract_features(pixels)                       # (B, 2048)
        txt_feat = self.text_model.extract_features(input_ids, attention_mask)     # (B, 128)
        img_probs = self.image_model.classify(img_feat).softmax(dim=1)              # (B, 101)
        txt_probs = self.text_model.classify(txt_feat).softmax(dim=1)               # (B, 101)
        # Same inputs as in training: softmax vectors for late fusion, features for early fusion.
        fusion_inputs = (img_probs, txt_probs) if self.is_late else (img_feat, txt_feat)
        fus_probs = self.fusion_model(*fusion_inputs).softmax(dim=1)                # (B, 101)

        return [
            {
                "title": clean[i],
                "image": self._top_k(img_probs[i], top_k),
                "text": self._top_k(txt_probs[i], top_k),
                "fusion": self._top_k(fus_probs[i], top_k),
            }
            for i in range(len(images))
        ]

    def _top_k(self, probs: torch.Tensor, k: int) -> list[tuple[str, float]]:
        """(101,) probabilities -> [(class name, probability), ...] for the k most likely classes."""
        values, indices = probs.topk(k)
        return [(self.class_names[i], round(v, 4)) for v, i in zip(values.tolist(), indices.tolist())]


def load_test_samples(
    splits_dir: str, data_dir: str, sample_ids: list[str], n_random: int, seed: int
) -> list[dict]:
    """Test samples from the volume: the given sample_ids, or n_random random ones (fixed seed).

    Out: [{sample_id, title, true, image (raw bytes)}], in the requested order.
    """
    test = pd.read_csv(os.path.join(splits_dir, "test.csv"))     # columns: image, title, label
    if sample_ids:
        unknown = sorted(set(sample_ids) - set(test["image"]))
        if unknown:
            raise ValueError(f"not in the test split: {unknown}")
        rows = test.set_index("image").loc[sample_ids].reset_index()
    else:
        rows = test.sample(n_random, random_state=seed)
    samples = []
    for _, row in rows.iterrows():
        path = os.path.join(data_dir, "images", "test", row["label"], row["image"])
        with open(path, "rb") as f:
            samples.append({"sample_id": row["image"], "title": row["title"] if isinstance(row["title"], str) else "",
                            "true": row["label"], "image": f.read()})
    return samples
