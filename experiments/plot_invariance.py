import torch
import numpy as np
import matplotlib.pyplot as plt

from data.synthetic import generate_point_cloud
from data.symmetry import permute_point_cloud
from models.encoder import PointCloudEncoder
from models.group_average import group_averaged_embedding

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Generate data
N_POINTS = 10
pc = generate_point_cloud(n_points=N_POINTS)
pc_tensor = torch.tensor(pc, dtype=torch.float32)

# Encoder
encoder = PointCloudEncoder(n_points=N_POINTS)

# Store distances
baseline_distances = []
invariant_distances = []

# Original embeddings
z_base_orig = encoder(pc_tensor)
z_inv_orig = group_averaged_embedding(pc, encoder)

# Apply permutations
for _ in range(10):
    pc_perm = permute_point_cloud(pc)
    pc_perm_tensor = torch.tensor(pc_perm, dtype=torch.float32)

    # Baseline
    z_base_perm = encoder(pc_perm_tensor)
    baseline_distances.append(
        torch.norm(z_base_orig - z_base_perm).item()
    )

    # Invariant
    z_inv_perm = group_averaged_embedding(pc_perm, encoder)
    invariant_distances.append(
        torch.norm(z_inv_orig - z_inv_perm).item()
    )

# Plot
plt.figure()
plt.plot(baseline_distances, label="Baseline Encoder")
plt.plot(invariant_distances, label="Group-Averaged Encoder")
plt.xlabel("Permutation Index")
plt.ylabel("Embedding Distance")
plt.title("Effect of Permutation Symmetry on Learned Representations")
plt.legend()
plt.tight_layout()

plt.savefig("results/embedding_invariance.png")
plt.show()
