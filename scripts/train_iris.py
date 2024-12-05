import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from alibi_testing.data import get_iris_data


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
    model.fit(x_train, y_train, batch_size=128, epochs=500)
    return model


def run_ae():
    data = get_iris_data()
    x_train, y_train = data['X_train'], data['y_train']
    ae, enc, dec = ae_model()
    ae.fit(x_train, x_train, batch_size=32, epochs=100)
    return ae, enc


def run_model(name):
    if name == 'ffn':
        model = run_ffn()
        return model
    elif name == 'ae':
        ae, enc = run_ae()
        return ae, enc
    else:
        raise ValueError(f'Unknown model: {name}')


def saved_name(model_name):
    data = 'iris'
    framework = 'tf'
    tf_ver = tf.__version__
    if int(tf_ver[0]) < 2:
        suffix = '.h5'
    else:
        suffix = '.keras'
    tf_ver = framework + tf_ver

    return '-'.join((data, model_name, tf_ver)) + suffix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of the model to train')

    args = parser.parse_args()
    models = run_model(args.model)

    if args.model == 'ffn':
        name = saved_name(args.model)
        models.save(name)
    elif args.model == 'ae':
        ae_name = saved_name(args.model)
        enc_name = saved_name('enc')
        models[0].save(ae_name)
        models[1].save(enc_name)
