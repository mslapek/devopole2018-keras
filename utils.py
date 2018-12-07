import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris


def raw_to_data(raw, y_name):
    """Converts sklearn datasets to DataFrame."""
    data = pd.DataFrame({
        n: v for n, v in zip(raw.feature_names, raw.data.T)
    })
    data[y_name] = raw.target
    return data


def get_california_data():
    """
    Regression example for California Housing dataset
    (see https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset).
    """

    california_data = raw_to_data(
        fetch_california_housing(),
        'MedValue'
    )

    np.random.seed(0)
    california_data = california_data.sample(frac=1)
    return california_data


def get_iris_data():
    raw = load_iris()
    data = raw_to_data(
        raw, 'class'
    )
    data['family'] = raw.target_names[data['class']]
    return data


def make_tomato(size):
    """
    Simulated regression example.
    """

    np.random.seed(0)
    x = np.random.normal(size=size)
    tomato_price = 5 + x
    ketchup_price = (2 * np.sin(2 * x) + x +
                     np.random.normal(scale=0.5, size=size)) / 2 + 3

    return pd.DataFrame({
        'tomato_price': tomato_price,
        'ketchup_price': ketchup_price
    })
