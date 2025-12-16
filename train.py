import torch
import numpy as np

from data.synthetic import generate_point_cloud
from data.symmetry import permute_point_cloud
from models.encoder import PointCloudEncoder

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Generate one point cloud
pc = generate_point_cloud(n_points=10)
pc_tensor = torch.tensor(pc, dtype=torch.float32)

# Initialize encoder
encoder = PointCloudEncoder()

# Original embedding
z_original = encoder(pc_tensor)

print("Baseline encoder instability test:\n")

# Apply multiple permutations
for i in range(5):
    pc_perm = permute_point_cloud(pc)
    pc_perm_tensor = torch.tensor(pc_perm, dtype=torch.float32)

    z_perm = encoder(pc_perm_tensor)

    distance = torch.norm(z_original - z_perm).item()
    print(f"Permutation {i+1} | Embedding distance: {distance:.6f}")
