import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from alibi.datasets import fetch_adult, fetch_movie_sentiment


def get_iris_data(seed=42):
    """
    Load the Iris dataset.
    """
    data = load_iris()
    X, y = data.data, data.target
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # scale dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    return {
        'X_train': x_train,
        'y_train': y_train,
        'X_test': x_test,
        'y_test': y_test,
        'preprocessor': None,
        'metadata': {
            'feature_names': data.feature_names,
            'name': 'iris'
        }
    }


def get_mnist_data():
    """
    Load the MNIST dataset.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    xmin, xmax = -.5, .5
    x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
    x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin

    return {
        'X_train': x_train,
        'y_train': y_train,
        'X_test': x_test,
        'y_test': y_test,
        'preprocessr': None,
        'metadata': {
            'name': 'mnist'
        }
    }


def get_movie_sentiment_data(seed=42):
    """
    Load and prepare movie sentiment dataset.
    """

    movies = fetch_movie_sentiment()
    data = movies.data
    labels = movies.target
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed)
    y_train = np.array(y_train)
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(x_train)

    return {
        'X_train': x_train,
        'y_train': y_train,
        'X_test': x_test,
        'y_test': y_test,
        'preprocessor': vectorizer,
        'metadata': {'name': 'movie_sentiment'},
    }


def get_adult_data(seed=42):
    """
    Load the Adult dataset.
    """
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

    return {
        'X_train': x_train,
        'y_train': y_train,
        'X_test': x_test,
        'y_test': y_test,
        'preprocessr': None,
        'metadata': {
            'name': 'adult'
        }
    }
