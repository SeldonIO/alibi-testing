import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from alibi_testing.data import get_iris_data
from utils import validate_args, disable_v2_behavior, save_model_tf


def ffn_model():
    x_in = Input(shape=(4,))
    x = Dense(10, activation='relu')(x_in)
    x_out = Dense(3, activation='softmax')(x)

    ffn = Model(inputs=x_in, outputs=x_out)
    ffn.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return ffn


def ae_model():
    # encoder
    x_in = Input(shape=(4,))
    x = Dense(5, activation='relu')(x_in)
    encoded = Dense(2, activation=None)(x)
    encoder = Model(inputs=x_in, outputs=encoded)

    # decoder
    dec_in = Input(shape=(2,))
    x = Dense(5, activation='relu')(dec_in)
    decoded = Dense(4, activation=None)(x)
    decoder = Model(inputs=dec_in, outputs=decoded)

    # autoencoder = encoder + decoder
    x_out = decoder(encoder(x_in))
    autoencoder = Model(inputs=x_in, outputs=x_out)
    autoencoder.compile(loss='mse', optimizer='adam')

    return autoencoder, encoder, decoder


def run_ffn():
    data = get_iris_data()
    x_train, y_train = data['X_train'], data['y_train']
    model = ffn_model()
    model.fit(x_train, y_train, batch_size=128, epochs=5)
    return model


def run_ae():
    data = get_iris_data()
    x_train, _ = data['X_train'], data['y_train']
    ae, enc, _ = ae_model()
    ae.fit(x_train, x_train, batch_size=32, epochs=100)
    return ae, enc


def run_model(args):
    # disable v2 behavior if necessary
    disable_v2_behavior(args)

    if args.model == 'ffn':
        return run_ffn()
    
    if args.model == 'ae':
        return run_ae()
    
    raise ValueError(f'Unknown model: {args.model}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Name of the model to train')
    parser.add_argument('--format', type=str, choices=["h5", "keras"])
    args = parser.parse_args()

    # validate args combination
    validate_args(args)

    # train the model
    models = run_model(args)

    # save the models
    kwargs = {
        "data": "iris",
        "framework": "tf",
        "version": tf.__version__
    }
    
    if args.model == 'ffn':
        save_model_tf(models, args, model_name=args.model, **kwargs) 
    elif args.model == 'ae':
        save_model_tf(models[0], args, model_name=args.model, **kwargs)
        save_model_tf(models[1], args, model_name="enc", **kwargs)
