import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    pts = np.asarray(points, dtype=float)
    query = np.asarray(query_point, dtype=float)

    distances = np.linalg.norm(pts - query, axis=1)

    nearest_neighbors = np.argsort(distances, kind='stable')[:k]

    return [points[i] for i in nearest_neighbors]

# class SimpleLSH:
#     def __init__(self, dim: int, r: float = 2.0, seed: int = 42):
#         rng = np.random.default_rng(seed)

#         self.w = rng.normal(0, 1, size=dim)
#         self.r = r
    
#     def hash_points(self, X: np.ndarray) -> np.ndarray:
#         projections = X @ self.w
#         return np.floor(projections / self.r).astype(int)

# rng = np.random.default_rng(42)
# dim = 10
# n_pairs = 5000

# X1_close = rng.normal(0, 5, size=(n_pairs, dim))
# X2_close = X1_close + rng.normal(0, 0.2, size=(n_pairs, dim))

# X1_far = rng.normal(0, 5, size=(n_pairs, dim))
# X2_far = rng.normal(0, 5, size=(n_pairs, dim))

# lsh = SimpleLSH(dim=dim, r=2.0, seed=42)

# close_collisions = np.mean(lsh.hash_points(X1_close) == lsh.hash_points(X2_close))
# far_collisions = np.mean(lsh.hash_points(X1_far) == lsh.hash_points(X2_far))

# print(
#     "Среднее расстояние (близкие):"
#     f" {np.mean(np.linalg.norm(X1_close - X2_close, axis=1)):.2f}"
# )
# print(
#     "Среднее расстояние (далёкие):"
#     f" {np.mean(np.linalg.norm(X1_far - X2_far, axis=1)):.2f}\n"
# )
# print(
#     "Вероятность попасть в одну корзину для близких точек:"
#     f" {close_collisions:.2%}"
# )
# print(
#     "Вероятность попасть в одну корзину для далеких точек:"
#     f" {far_collisions:.2%}"
# )
