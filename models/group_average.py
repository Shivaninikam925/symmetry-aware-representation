import torch
from data.symmetry import permute_point_cloud

def group_averaged_embedding(x_np, encoder, num_permutations=5):
    """
    x_np: numpy array of shape (N, 2)
    encoder: PointCloudEncoder
    returns: torch Tensor of shape (32,)
    """
    embeddings = []

    for _ in range(num_permutations):
        x_perm = permute_point_cloud(x_np)
        x_tensor = torch.tensor(x_perm, dtype=torch.float32)
        z = encoder(x_tensor)
        embeddings.append(z)

    return torch.stack(embeddings).mean(dim=0)
