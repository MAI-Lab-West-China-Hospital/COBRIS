# models/unilateral_resnet.py

import torch
import torch.nn as nn
from monai.networks.nets import resnet18, resnet50

class UnilateralResNetCls(nn.Module):
    """
    Standard 3D ResNet Classifier.
    Structure:
      1. Encoder: 3D ResNet -> returns feature vector [B, 256]
      2. Classifier: [B, 256] -> [B, 128] -> [B, 2]
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        feature_dim: int = 256,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()

        if backbone == "resnet50":
            self.encoder = resnet50(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=feature_dim,
            )
        elif backbone == "resnet18":
            self.encoder = resnet18(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=feature_dim,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return logits
