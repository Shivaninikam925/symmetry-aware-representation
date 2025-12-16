import torch
import numpy as np

from data.synthetic import generate_point_cloud
from models.encoder import PointCloudEncoder

# Generate a sample point cloud
pc = generate_point_cloud()
pc_tensor = torch.tensor(pc, dtype=torch.float32)

# Initialize encoder
encoder = PointCloudEncoder()

# Get embedding
embedding = encoder(pc_tensor)

print("Embedding shape:", embedding.shape)
print("Embedding:", embedding)
