from pathlib import Path
import tensorflow as tf
import torch

BASE_PATH = Path(__file__).parent
MODEL_PATH = (BASE_PATH / 'models').resolve()
models = MODEL_PATH.glob('*')
MODEL_REGISTRY = ([p.name for p in models])


def load(name: str):
    path = (MODEL_PATH / f'{name}').resolve()
    try:
        # tensorflow
        model = tf.keras.models.load_model(path)
    except OSError:
        # pytorch
        model = torch.load(path)
    return model
