import torch

def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """
    data = torch.as_tensor(data, dtype=torch.float64)

    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0, correction=0)
    std_safe = torch.where(std == 0, torch.tensor(1.0, dtype=data.dtype), std)
    standardized_data = (data - mean) / std_safe

    cov_matrix = torch.cov(standardized_data.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_k_vectors = eigenvectors[:, sorted_indices[:k]]

    for j in range(k):
        vec = top_k_vectors[:, j]
        non_zero_indices = torch.where(torch.abs(vec) > 1e-10)[0]
        if len(non_zero_indices) > 0:
            first_idx = non_zero_indices[0]
            if vec[first_idx] < 0:
                top_k_vectors[:, j] = -vec

    return torch.round(top_k_vectors, decimals=4)
