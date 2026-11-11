"""Image model: ImageNet-pretrained InceptionV3 with the Table I head, for UPMC Food-101.

In:  images (B, 3, 299, 299), ImageNet-normalized (see src/data.py)
Out: logits (B, 101); extract_features gives the 2048-d pooled features for fusion.
"""
import torch
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

# InceptionV3 blocks in forward order, split where the auxiliary classifier branches off
# (it reads the output of Mixed_6e).
BLOCKS_BEFORE_AUX = (
    "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1",
    "Conv2d_3b_1x1", "Conv2d_4a_3x3", "maxpool2",
    "Mixed_5b", "Mixed_5c", "Mixed_5d",
    "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e",
)
BLOCKS_AFTER_AUX = ("Mixed_7a", "Mixed_7b", "Mixed_7c")


class ImageClassifier(nn.Module):
    """InceptionV3 backbone (up to Mixed_7c) + AvgPool -> Dropout -> Flatten -> Linear (Table I)."""

    def __init__(
        self,
        num_classes: int,
        pool_size: int,
        dropout: float,
        use_aux_logits: bool,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        # aux_logits=True is required to load the pretrained weights; we decide below whether to keep it.
        backbone = inception_v3(weights=weights, aux_logits=True)
        self.feature_dim = backbone.fc.in_features  # 2048 = channels of Mixed_7c

        # Remove torchvision's own head (global pool, dropout 0.5, FC 2048->1000); ours replaces it.
        backbone.avgpool, backbone.dropout, backbone.fc = nn.Identity(), nn.Identity(), nn.Identity()

        if use_aux_logits:
            aux_in = backbone.AuxLogits.fc.in_features  # 768
            backbone.AuxLogits.fc = nn.Linear(aux_in, num_classes)  # 1000 ImageNet -> 101 food classes
        else:
            backbone.AuxLogits = None
        self.backbone = backbone

        # --- Table I head ---
        self.pool = nn.AvgPool2d(pool_size)       # (B, 2048, 8, 8) -> (B, 2048, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()               # (B, 2048, 1, 1) -> (B, 2048)
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def _run_backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, 299, 299) -> Mixed_7c map (B, 2048, 8, 8) and Mixed_6e map (B, 768, 17, 17)."""
        # The pretrained weights expect inputs scaled to [-1, 1]; with pretrained weights torchvision
        # sets transform_input=True, and this converts our ImageNet-normalized input accordingly.
        x = self.backbone._transform_input(x)
        for name in BLOCKS_BEFORE_AUX:
            x = getattr(self.backbone, name)(x)
        mixed_6e = x                              # (B, 768, 17, 17)
        for name in BLOCKS_AFTER_AUX:
            x = getattr(self.backbone, name)(x)
        return x, mixed_6e                        # (B, 2048, 8, 8), (B, 768, 17, 17)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 299, 299) -> pooled features (B, 2048), before dropout. Used for fusion."""
        feature_map, _ = self._run_backbone(x)
        return self.flatten(self.pool(feature_map))  # (B, 2048)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Pooled features (B, 2048) -> logits (B, 101): the end of forward (dropout is a no-op in eval)."""
        return self.fc(self.dropout(features))

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(B, 3, 299, 299) -> (logits (B, 101), aux_logits (B, 101) in training, else None)."""
        feature_map, mixed_6e = self._run_backbone(x)
        x = self.pool(feature_map)                # (B, 2048, 1, 1)
        x = self.dropout(x)
        x = self.flatten(x)                       # (B, 2048)
        logits = self.fc(x)                       # (B, 101)

        aux_logits = None
        if self.training and self.backbone.AuxLogits is not None:
            aux_logits = self.backbone.AuxLogits(mixed_6e)  # (B, 101)
        return logits, aux_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 299, 299) -> logits (B, 101). Use forward_with_aux in training to get the aux loss."""
        return self.forward_with_aux(x)[0]


def build_image_model(config: dict) -> ImageClassifier:
    """Build the ImageClassifier from the architecture keys of configs/image.yaml."""
    return ImageClassifier(
        num_classes=config["num_classes"],
        pool_size=config["pool_size"],
        dropout=config["dropout"],
        use_aux_logits=config["use_aux_logits"],
        pretrained=config["pretrained"],
    )


def shape_check(config: dict) -> None:
    """Run a dummy (2, 3, 299, 299) batch through the model and print every output shape."""
    model = build_image_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ImageClassifier: {n_params / 1e6:.1f}M parameters")

    dummy = torch.randn(2, 3, 299, 299)
    model.eval()
    with torch.no_grad():
        print("eval  logits         ", tuple(model(dummy).shape))
        print("eval  features       ", tuple(model.extract_features(dummy).shape))
        print("eval  aux_logits     ", model.forward_with_aux(dummy)[1])

    model.train()
    with torch.no_grad():
        logits, aux_logits = model.forward_with_aux(dummy)
    print("train logits         ", tuple(logits.shape))
    print("train aux_logits     ", None if aux_logits is None else tuple(aux_logits.shape))
