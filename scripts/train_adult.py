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


def run_model(name):
    data = get_adult_data(categorical_target=True)
    pre = data['preprocessor']
    x_train, x_test = pre.transform(data['X_train']), pre.transform(data['X_test'])
    y_train, y_test = data['y_train'], data['y_test']
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
