import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = []
    total_squared_error = 0.0

    for feature_vec, label in zip(features, labels):
        z = sum(x*w for x, w in zip(feature_vec, weights)) + bias

        prob = 1.0 / (1.0 + math.exp(-z))
        probabilities.append(round(prob, 4))

        total_squared_error += (prob - label) ** 2
    
    mse = round(total_squared_error / len(labels), 4)

    return probabilities, mse