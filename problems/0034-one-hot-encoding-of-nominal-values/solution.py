import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def to_categorical(x, n_col=None):
    if n_col is None:
        n_col = np.max(x) + 1
    
    cat = pd.Categorical(x, categories=range(n_col))

    return pd.get_dummies(cat, dtype=float).to_numpy()

    # encoder = OneHotEncoder(categories=[range(n_col)], sparse_output=False)

    # return encoder.fit_transform(x.reshape(-1, 1))