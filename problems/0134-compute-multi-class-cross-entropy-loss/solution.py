import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    n = predicted_probs.shape[0]
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    true_labels = np.asarray(true_labels, dtype=float)

    clipped_probs = np.clip(predicted_probs, epsilon, 1.0 - epsilon)

    loss = (-1 / n) * np.sum(true_labels * np.log(clipped_probs))

    return loss