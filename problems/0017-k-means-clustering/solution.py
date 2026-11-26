import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    pts = np.asarray(points, dtype=float)
    centroids = np.asarray(initial_centroids, dtype=float)

    for _ in range(max_iterations):
        distances = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = pts[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break
        
        centroids = new_centroids
    
    final_centroids = [tuple(np.round(c, 4).tolist()) for c in centroids]

    return final_centroids