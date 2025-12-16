import torch
import torch.nn as nn

class PointCloudEncoder(nn.Module):
    def __init__(self, n_points=10):
        super().__init__()
        self.n_points = n_points

        self.mlp = nn.Sequential(
            nn.Linear(2 * n_points, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        """
        x: Tensor of shape (N, 2)
        returns: Tensor of shape (32,)
        """
        x_flat = x.view(-1)   # (2N,)
        z = self.mlp(x_flat)
        return z
