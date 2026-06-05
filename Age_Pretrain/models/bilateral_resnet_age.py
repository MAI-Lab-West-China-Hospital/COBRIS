# models/bilateral_resnet_age.py

import torch
import torch.nn as nn
from monai.networks.nets import resnet18, resnet34, resnet50, resnet101, seresnet50, seresnet101


class BilateralResNetAge(nn.Module):
    """
    Bilateral 3D ResNet-based regressor.

    Inputs:
        LEFT, RIGHT: [B, 1, 256, 256, 64]  (3D NIfTI volumes after transforms)
    Output:
        age: [B]  (regression)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        feature_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()

        if backbone == "resnet50":
            self.encoder = resnet50(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=feature_dim,  # acts as feature extractor head
            )
        elif backbone == "resnet18":
            self.encoder = resnet18(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=feature_dim,
            )
        elif backbone == "resnet34":
            self.encoder = resnet34(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=feature_dim,
            )
        elif backbone == "resnet101":
            self.encoder = resnet101(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=feature_dim,
            )
        elif backbone == "seresnet50":
            self.encoder = seresnet50(
                spatial_dims=3,
                in_channels=1,
                num_classes=feature_dim,
            )
        elif backbone == "seresnet101":
            self.encoder = seresnet101(
                spatial_dims=3,
                in_channels=1,
                num_classes=feature_dim,
            )       
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        left, right: [B, 1, H, W, D]
        returns: age [B]
        """
        feat_left = self.encoder(left)      # [B, feature_dim]
        feat_right = self.encoder(right)    # [B, feature_dim]
        feat = torch.cat([feat_left, feat_right], dim=1)  # [B, 2*feature_dim]
        age = self.regressor(feat).squeeze(1)             # [B]
        return age

