import argparse

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from alibi_testing.data import get_adult_data
from utils import validate_args, disable_v2_behavior, save_model


def ffn_model():
    x_in = Input(shape=(57,))
    x = Dense(60, activation='relu')(x_in)
    x = Dense(60, activation='relu')(x)
    x_out = Dense(2, activation='softmax')(x)

    ffn = Model(inputs=x_in, outputs=x_out)
    ffn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return ffn


def run_model(args):
    # disable v2 behavior if necessary
    disable_v2_behavior(args)

    data = get_adult_data(categorical_target=True)
    pre = data['preprocessor']
    x_train, x_test = pre.transform(data['X_train']), pre.transform(data['X_test'])
    y_train, y_test = data['y_train'], data['y_test']
    
    model = globals()[f'{args.model}_model']()
    model.fit(x_train, y_train, batch_size=128, epochs=5)
    model.evaluate(x_test, y_test)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Name of the model to train')
    parser.add_argument('--format', type=str, choices=["h5", "keras"])
    args = parser.parse_args()

    # validate args combinations
    validate_args(args)

    # train the model
    model = run_model(args)

    # save the trained moel
    save_model(
        model=model, 
        args=args,
        model_name=args.model,
        data="adult", 
        framework="tf", 
        version=tf.__version__
    )
