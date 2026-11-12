"""Fusion models over the cached features of src/extract.py.

LateFusion (Table III): merge the image and text softmax vectors, then an MLP.
    In:  image_probs (B, 101), text_probs (B, 101) from /ckpt/features/{split}.pt
EarlyFusion (Table IV): project the image features to the text size, concatenate, then an MLP.
    In:  image_features (B, 2048), text_features (B, 128) from /ckpt/features/{split}.pt
Out (both): logits (B, 101)
"""
import torch
from torch import nn

MERGE_MODES = ("concat", "sum", "max", "weighted")


class LateFusion(nn.Module):
    """Merge two (B, C) softmax vectors, then Linear -> ReLU -> Dropout blocks -> Linear(C)."""

    def __init__(
        self,
        num_classes: int,
        merge: str,
        hidden_dims: list[int],
        dropout: float,
        alpha_logit_init: float = 0.0,
    ) -> None:
        super().__init__()
        if merge not in MERGE_MODES:
            raise ValueError(f"merge must be one of {MERGE_MODES}, got {merge!r}")
        self.merge = merge

        # Only "weighted" has a learnable merge parameter. Store the raw (unbounded) value and
        # apply a sigmoid in forward, so alpha always stays in (0, 1) and needs no clamping.
        self.alpha_logit = (
            nn.Parameter(torch.tensor(alpha_logit_init)) if merge == "weighted" else None
        )

        # concat puts the two vectors side by side (2C); the other modes keep C.
        in_dim = 2 * num_classes if merge == "concat" else num_classes   # 202 or 101
        self.mlp = build_mlp(in_dim, hidden_dims, num_classes, dropout)

    @property
    def alpha(self) -> torch.Tensor | None:
        """Weight on the image vector in "weighted" mode (text gets 1 - alpha). None otherwise."""
        return None if self.alpha_logit is None else torch.sigmoid(self.alpha_logit)

    def merge_inputs(self, image_probs: torch.Tensor, text_probs: torch.Tensor) -> torch.Tensor:
        """(B, C), (B, C) -> merged (B, 2C) for concat, otherwise (B, C)."""
        if self.merge == "concat":
            return torch.cat([image_probs, text_probs], dim=1)           # (B, 202)
        if self.merge == "sum":
            return image_probs + text_probs                              # (B, 101)
        if self.merge == "max":
            return torch.maximum(image_probs, text_probs)                # (B, 101), per class
        alpha = self.alpha                                               # scalar in (0, 1)
        return alpha * image_probs + (1 - alpha) * text_probs            # (B, 101)

    def forward(self, image_probs: torch.Tensor, text_probs: torch.Tensor) -> torch.Tensor:
        """(B, C), (B, C) softmax vectors -> logits (B, C)."""
        return self.mlp(self.merge_inputs(image_probs, text_probs))


def build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float) -> nn.Sequential:
    """[Linear -> ReLU -> Dropout] for each hidden size, then a final Linear to out_dim."""
    layers: list[nn.Module] = []
    for hidden in hidden_dims:                    # Table III: 256, then 128
        layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, out_dim))     # (B, 128) -> (B, 101), no softmax: CE takes logits
    return nn.Sequential(*layers)


def build_late_fusion(config: dict) -> LateFusion:
    """Build LateFusion from configs/late_fusion.yaml (flat keys)."""
    return LateFusion(
        num_classes=config["num_classes"],
        merge=config["merge"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        alpha_logit_init=config["alpha_logit_init"],
    )


class EarlyFusion(nn.Module):
    """Image branch (Dropout -> Linear -> ReLU) to text_dim, concat with text, then a 2-layer head."""

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        image_proj_dim: int,
        image_dropout: float,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        # --- Table IV, image branch: 2048 -> 128 ---
        self.image_branch = nn.Sequential(
            nn.Dropout(image_dropout),
            nn.Linear(image_dim, image_proj_dim),   # (B, 2048) -> (B, 128)
            nn.ReLU(),
        )
        # --- Table IV, head over the concatenation ---
        self.head = nn.Sequential(
            nn.Linear(image_proj_dim + text_dim, hidden_dim),   # (B, 256) -> (B, 256)
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),                 # (B, 256) -> (B, 101), logits
        )

    def fuse(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """(B, 2048), (B, 128) -> fused vector (B, 256) = [projected image, text]."""
        image_proj = self.image_branch(image_features)          # (B, 128)
        return torch.cat([image_proj, text_features], dim=1)    # (B, 256)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """(B, 2048), (B, 128) cached features -> logits (B, 101)."""
        return self.head(self.fuse(image_features, text_features))


def build_early_fusion(config: dict) -> EarlyFusion:
    """Build EarlyFusion from configs/early_fusion.yaml (flat keys)."""
    return EarlyFusion(
        image_dim=config["image_dim"],
        text_dim=config["text_dim"],
        image_proj_dim=config["image_proj_dim"],
        image_dropout=config["image_dropout"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
    )


def early_shape_check(config: dict) -> None:
    """Dummy cached features through EarlyFusion, printing the shape after every step."""
    batch = config["shape_check_batch_size"]
    image_features = torch.randn(batch, config["image_dim"]).relu()   # (B, 2048), >= 0 like pooled ReLU maps
    text_features = torch.randn(batch, config["text_dim"]).tanh()     # (B, 128), in (-1, 1) like an LSTM h_n
    model = build_early_fusion(config).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EarlyFusion: {n_params:,} params")

    with torch.no_grad():
        image_proj = model.image_branch(image_features)
        fused = torch.cat([image_proj, text_features], dim=1)
        hidden = model.head[1](model.head[0](fused))
        logits = model(image_features, text_features)
    for name, t in [("image_features", image_features), ("text_features", text_features),
                    ("image_proj", image_proj), ("fused (concat)", fused),
                    ("hidden", hidden), ("logits", logits)]:
        print(f"  {name:<15} {tuple(t.shape)}")


def shape_check(config: dict) -> None:
    """Build every merge mode and print the merged and output shapes for dummy softmax inputs."""
    batch, classes = config["shape_check_batch_size"], config["num_classes"]
    image_probs = torch.randn(batch, classes).softmax(dim=1)   # (B, 101), rows sum to 1
    text_probs = torch.randn(batch, classes).softmax(dim=1)    # (B, 101)

    for merge in MERGE_MODES:
        model = build_late_fusion({**config, "merge": merge}).eval()
        n_params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            merged = model.merge_inputs(image_probs, text_probs)
            logits = model(image_probs, text_probs)
        alpha = "" if model.alpha is None else f" | alpha {model.alpha.item():.2f}"
        print(f"{merge:>8}: merged {tuple(merged.shape)} -> logits {tuple(logits.shape)}"
              f" | {n_params:,} params{alpha}")


if __name__ == "__main__":
    import yaml

    with open("configs/late_fusion.yaml") as f:
        shape_check(yaml.safe_load(f))
    with open("configs/early_fusion.yaml") as f:
        early_shape_check(yaml.safe_load(f))
