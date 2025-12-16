import numpy as np

def generate_point_cloud(n_points=10):
    """
    Generates a single 2D point cloud.
    Returns shape: (n_points, 2)
    """
    return np.random.randn(n_points, 2)


def generate_dataset(num_samples=500, n_points=10):
    """
    Generates a dataset of point clouds.
    Returns shape: (num_samples, n_points, 2)
    """
    data = []
    for _ in range(num_samples):
        pc = generate_point_cloud(n_points)
        data.append(pc)
    return np.array(data)


if __name__ == "__main__":
    dataset = generate_dataset()
    print("Dataset shape:", dataset.shape)
