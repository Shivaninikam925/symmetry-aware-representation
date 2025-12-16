import numpy as np

def permute_point_cloud(point_cloud):
    """
    Applies a random permutation to a point cloud.
    Input shape: (N, 2)
    Output shape: (N, 2)
    """
    idx = np.random.permutation(len(point_cloud))
    return point_cloud[idx]


if __name__ == "__main__":
    pc = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    print("Original:")
    print(pc)

    print("\nPermuted:")
    print(permute_point_cloud(pc))
