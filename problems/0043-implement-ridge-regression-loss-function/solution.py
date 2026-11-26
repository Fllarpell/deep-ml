import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:

    X = np.asarray(X, dtype=float)
    y = np.asarray(y_true, dtype=float)
    w = np.asarray(w, dtype=float)

    y_pred = X @ w

    mse = np.mean((y - y_pred)**2)

    reg = alpha * np.sum(w**2)

    return float(mse + reg)