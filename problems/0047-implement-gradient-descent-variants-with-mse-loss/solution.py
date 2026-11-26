import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):

    w = np.asarray(weights, dtype=float).copy()


    m = X.shape[0]  # берем из кортежа (строки, столбцы) — кол-во строк

    for _ in range(n_epochs):
        if method == 'batch':
            grad = (2 / m) * X.T @ (X @ w - y)
            w -= learning_rate * grad

        elif method == 'mini_batch':
            for start_idx in range(0, m, batch_size):
                X_b = X[start_idx:start_idx+batch_size]
                y_b = y[start_idx:start_idx+batch_size]
                k = X_b.shape[0]
                grad = (2 / k) * X_b.T @ (X_b @ w - y_b)
                w -= learning_rate * grad
        
        elif method == 'stochastic':
            for i in range(m):
                X_i = X[i:i+1]
                y_i = y[i:i+1]
                grad = 2 * X_i.T @ (X_i @ w - y_i)
                w -= learning_rate * grad

    return w
