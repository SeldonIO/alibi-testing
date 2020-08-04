import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from alibi.datasets import fetch_adult, fetch_movie_sentiment
from alibi.utils.mapping import ord_to_ohe


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


def get_boston_data(seed=42):
    """
    Load the Boston housing dataset.
    """
    dataset = load_boston()
    data = dataset.data
    labels = dataset.target
    feature_names = dataset.feature_names
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': None,
        'metadata': {
            'feature_names': feature_names,
            'name': 'boston'}
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


def get_adult_data(seed=42, categorical_target=False):
    """
    Load and preprocess the Adult dataset.
    """

    # load raw data
    adult = fetch_adult()
    data = adult.data
    target = adult.target
    feature_names = adult.feature_names
    category_map = adult.category_map

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)

    # Create feature transformation pipeline
    ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
    ordinal_transformer = MinMaxScaler(feature_range=(-1.0, 1.0))

    categorical_features = list(category_map.keys())
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', ordinal_transformer, ordinal_features)
        ]
    )
    preprocessor.fit(X_train)

    if categorical_target:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    # calculate categorical variable embeddings with the modified feature columns
    cat_vars_ord = {}
    for ix, (key, val) in enumerate(category_map.items()):
        cat_vars_ord[ix] = len(val)
    cat_vars_ohe = ord_to_ohe(cat_vars_ord)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'metadata': {
            'feature_names': feature_names,
            'category_map': category_map,
            'cat_vars_ord': cat_vars_ord,
            'cat_vars_ohe': cat_vars_ohe,
            'name': 'adult'
        }
    }
