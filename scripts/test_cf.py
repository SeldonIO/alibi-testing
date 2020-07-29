import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import load_model
from alibi.explainers import CounterFactual
from train_mnist import mnist_data
import argparse


def setup_cf(args):
    (x_train, y_train), (x_test, y_test) = mnist_data()
    model = load_model(args.model)
    X = x_test[0].reshape((1,) + x_test[0].shape)

    shape = X.shape
    target_proba = 1.0
    tol = 0.01  # want counterfactuals with p(class)>0.99
    target_class = 'other'  # any class other than 7 will do
    max_iter = 1000
    lam_init = 1e-1
    max_lam_steps = 10
    learning_rate_init = 0.1
    feature_range = (x_train.min(), x_train.max())

    cf = CounterFactual(model, shape=shape, target_proba=target_proba, tol=tol,
                        target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                        max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                        feature_range=feature_range)
    return cf, X


def run_cf(args):
    print(f'TF version: {tf.__version__}')
    cf, X = setup_cf(args)
    exp = cf.explain(X)
    cf_class = exp.cf['class']
    cf_proba = exp.cf['proba'][0][cf_class]
    print(f'Original prediction: {exp.orig_class}: {exp.orig_proba}')
    print(f'CF prediction: {cf_class}: {cf_proba}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='h5 TensorFlow or Keras MNIST model.')
    args = parser.parse_args()
    run_cf(args)
