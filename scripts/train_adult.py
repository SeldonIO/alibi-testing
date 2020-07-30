import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from alibi.datasets import fetch_adult


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


def ffn_model():
    x_in = Input(shape=(57,))
    x = Dense(60, activation='relu')(x_in)
    x = Dense(60, activation='relu')(x)
    x_out = Dense(2, activation='softmax')(x)

    ffn = Model(inputs=x_in, outputs=x_out)
    ffn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return ffn


def run_model(name):
    (x_train, y_train), (x_test, y_test) = adult_data()
    model = globals()[f'{name}_model']()
    model.fit(x_train, y_train, batch_size=128, epochs=5)
    model.evaluate(x_test, y_test)
    return model


def saved_name(model_name):
    data = 'adult'
    framework = 'tf'
    tf_ver = tf.__version__
    if int(tf_ver[0]) < 2:
        suffix = '.h5'
    else:
        suffix = ''
    tf_ver = framework + tf_ver

    return '-'.join((data, model_name, tf_ver)) + suffix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of the model to train')

    args = parser.parse_args()
    model = run_model(args.model)
    name = saved_name(args.model)
    model.save(name)
