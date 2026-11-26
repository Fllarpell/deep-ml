import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(weights, dtype=float)
    
    y_pred = X @ w + bias

    probabilities = 1.0 / (1.0 + np.exp(-y_pred))

    return (probabilities >= 0.5).astype(int)