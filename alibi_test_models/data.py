import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from alibi.datasets import fetch_adult


def iris_data(seed=42):
    X, y = load_iris(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # scale dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    return (x_train, y_train), (x_test, y_test)


def adult_data(seed=42):
    adult = fetch_adult()

    X = adult.data
    X_ord = np.c_[X[:, 1:8], X[:, 11], X[:, 0], X[:, 8:11]]
    y = adult.target

    # scale numerical features
    X_num = X_ord[:, -4:].astype(np.float32, copy=False)
    xmin, xmax = X_num.min(axis=0), X_num.max(axis=0)
    rng = (-1., 1.)
    X_num_scaled = (X_num - xmin) / (xmax - xmin) * (rng[1] - rng[0]) + rng[0]

    # OHE categorical features
    X_cat = X_ord[:, :-4].copy()
    ohe = OneHotEncoder()
    ohe.fit(X_cat)
    X_cat_ohe = ohe.transform(X_cat)

    # combine categorical and numerical data
    X_comb = np.c_[X_cat_ohe.todense(), X_num_scaled].astype(np.float32, copy=False)

    # split in train and test set
    x_train, x_test, y_train, y_test = train_test_split(X_comb, y, test_size=0.2, random_state=seed)

    assert x_train.shape[1] == 57
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)
