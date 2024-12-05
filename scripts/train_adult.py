import os
import argparse

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from alibi_testing.data import get_adult_data


def ffn_model():
    x_in = Input(shape=(57,))
    x = Dense(60, activation='relu')(x_in)
    x = Dense(60, activation='relu')(x)
    x_out = Dense(2, activation='softmax')(x)

    ffn = Model(inputs=x_in, outputs=x_out)
    ffn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return ffn


def get_legacy():
    return os.environ.get("TF_USE_LEGACY_KERAS", None)


def get_format(args):
    return "" if args.format is None else args.format


def run_model(args):
    is_legacy, format = get_legacy(), get_format(args)

    if is_legacy == "1" and format == "h5":
        tf.compat.v1.disable_v2_behavior()

    data = get_adult_data(categorical_target=True)
    pre = data['preprocessor']
    x_train, x_test = pre.transform(data['X_train']), pre.transform(data['X_test'])
    y_train, y_test = data['y_train'], data['y_test']
    
    model = globals()[f'{args.model}_model']()
    model.fit(x_train, y_train, batch_size=128, epochs=5)
    model.evaluate(x_test, y_test)
    return model


def save_model(model, args):
    data, framework = 'adult', "tf"
    is_legacy, format = get_legacy(), get_format(args)
    
    tf_ver = framework + tf.__version__
    name = '-'.join((data, args.model, tf_ver)) + '.' + format
    kwargs = {"save_format": format} if is_legacy == "1" and format == "h5" else {}
    model.save(name, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Name of the model to train')
    parser.add_argument('--format', type=str, choices=["h5", "keras"])
    args = parser.parse_args()

    is_legacy, format = get_legacy(), get_format(args)
    if format == "keras" and is_legacy == "1":
        raise RuntimeError("Invalid format when using `TF_USE_LEGACY_KERAS`.")
    
    if is_legacy is None and format == "":
        raise RuntimeError("Invalid format. Expected `keras` format.")

    model = run_model(args)
    save_model(model, args)
