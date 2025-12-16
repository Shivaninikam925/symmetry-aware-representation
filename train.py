import torch
import numpy as np

from data.synthetic import generate_point_cloud
from data.symmetry import permute_point_cloud
from models.encoder import PointCloudEncoder
from models.group_average import group_averaged_embedding

# -----------------------------
# Reproducibility
# -----------------------------
np.random.seed(0)
torch.manual_seed(0)

# -----------------------------
# Generate one point cloud
# -----------------------------
N_POINTS = 10
pc = generate_point_cloud(n_points=N_POINTS)
pc_tensor = torch.tensor(pc, dtype=torch.float32)

# -----------------------------
# Initialize baseline encoder
# -----------------------------
encoder = PointCloudEncoder(n_points=N_POINTS)

# =====================================================
# BASELINE TEST: Order-sensitive encoder
# =====================================================
print("\nBaseline encoder instability test:\n")

z_original = encoder(pc_tensor)

for i in range(5):
    pc_perm = permute_point_cloud(pc)
    pc_perm_tensor = torch.tensor(pc_perm, dtype=torch.float32)

    z_perm = encoder(pc_perm_tensor)
    distance = torch.norm(z_original - z_perm).item()

    print(f"Permutation {i+1} | Embedding distance: {distance:.6f}")

# =====================================================
# SYMMETRY-AWARE TEST: Group-averaged encoder
# =====================================================
print("\nGroup-averaged encoder invariance test:\n")

z_inv_original = group_averaged_embedding(
    x_np=pc,
    encoder=encoder,
    num_permutations=5
)

for i in range(5):
    pc_perm = permute_point_cloud(pc)

    z_inv_perm = group_averaged_embedding(
        x_np=pc_perm,
        encoder=encoder,
        num_permutations=5
    )

    distance = torch.norm(z_inv_original - z_inv_perm).item()
    print(f"Permutation {i+1} | Embedding distance: {distance:.6f}")
