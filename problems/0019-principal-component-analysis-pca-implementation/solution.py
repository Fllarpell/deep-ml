import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    data = np.asarray(data, dtype = float)
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    standardized_data = (data - mean) / std_safe

    cov_matrix = np.cov(standardized_data, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1] # разворот индексов
    top_k_vectors = eigenvectors[:, sorted_indices[:k]]

    for j in range(k):
        vec = top_k_vectors[:, j]
        non_zero_indices = np.where(np.abs(vec) > 1e-10)[0]
        if len(non_zero_indices) > 0:
            first_idx = non_zero_indices[0]
            if vec[first_idx] < 0:
                top_k_vectors[:, j] = -vec

    return np.round(top_k_vectors, 4)
    
