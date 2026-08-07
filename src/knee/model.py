import timm
import torch
import torch.nn as nn


class KneeModel(nn.Module):
    """Phase 1 baseline: single 2D backbone applied per slice, mean-pooled
    across slices, 12 sigmoid heads (BCE loss applied outside the model on
    the returned logits). Input: [batch, n_slices, 3, H, W]."""

    def __init__(self, backbone_name: str = "efficientnet_b0", num_labels: int = 12, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_slices, channels, height, width = x.shape
        x = x.view(batch * n_slices, channels, height, width)
        features = self.backbone(x)
        features = features.view(batch, n_slices, -1).mean(dim=1)
        return self.head(features)
