import torch

def k_means_clustering(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float)
    
    for _ in range(max_iterations):
        distances = torch.cdist(points_t, centroids)

        labels = torch.argmin(distances, dim=1)

        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            cluster_points = points_t[labels == j]
            if cluster_points.shape[0] > 0:
                new_centroids[j] = cluster_points.mean(dim=0)
            else:
                new_centroids[j] = centroids[j]

        if torch.allclose(centroids, new_centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    final_centroids = [
        tuple(round(val, 4) for val in row) for row in centroids.tolist()
    ]

    return final_centroids
