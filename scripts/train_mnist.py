import argparse

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, Reshape

from alibi_testing.data import get_mnist_data
from utils import validate_args, disable_v2_behavior, save_model_tf


def cnn_model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x_in)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_out = Dense(10, activation='softmax')(x)

    cnn = Model(inputs=x_in, outputs=x_out)
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return cnn


def logistic_model():
    model = Sequential([
        Reshape((784,), input_shape=(28, 28, 1)),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def run_model(name):
    # disable v2 behavior if necessary
    disable_v2_behavior(args)

    data = get_mnist_data()
    x_train, y_train, x_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    model = globals()[f'{name}_model']()
    model.fit(x_train, y_train, batch_size=64, epochs=3)
    model.evaluate(x_test, y_test)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Name of the model to train')
    parser.add_argument('--format', type=str, choices=["h5", "keras"])
    args = parser.parse_args()

    # validate arguments combination
    validate_args(args)

    # train model
    model = run_model(args.model)
    
    # save model
    save_model_tf(
        model=model,
        args=args,
        model_name=args.model,
        data="mnist",
        framework="tf",
        version=tf.__version__
    )
