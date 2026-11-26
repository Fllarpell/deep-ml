import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n_samples = data.shape[0]
    
    if kernel == 'linear':
        K = data @ data.T

    elif kernel == 'rbf':
        sq_norm = np.sum(data**2, axis=1, keepdims=True)
        sq_dists = sq_norm + sq_norm.T - 2 * (data @ data.T)
        K = np.exp(-sq_dists / (2 * (sigma**2)))
    
    alphas = np.zeros(n_samples, dtype=float)
    bias = 0.0

    for t in range(1, iterations + 1):
        eta = 1.0 / (lambda_val * t)

        for i in range(n_samples):
            g_i = np.sum(alphas * labels * K[:, i]) + bias

            if labels[i] * g_i < 1.0:
                alphas[i] = alphas[i] + eta * (1.0 - lambda_val * alphas[i])
                bias = bias + eta * labels[i]
            else:
                alphas[i] = alphas[i] - eta * lambda_val * alphas[i]

    return alphas.tolist(), float(bias)
