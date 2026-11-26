def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    n_samples = len(vectors[0])

    means = [sum(feature) / n_samples for feature in vectors]

    centered = [
        [val - mean for val in feature] for feature, mean in zip(vectors, means)
    ]

    cov_matrix = [
        [
            sum(x * y for x, y in zip(centered[i], centered[j])) / (n_samples - 1)
            for j in range(n_features)
        ]
        for i in range(n_features)
    ]

    return cov_matrix
