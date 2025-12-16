import torch
import torch.nn as nn

class PointCloudEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        """
        x: Tensor of shape (N, 2)
        returns: Tensor of shape (32,)
        """
        h = self.mlp(x)          # (N, 32)
        z = h.mean(dim=0)        # (32,)
        return z
