"""
All models that are used for training.
"""
import os
import sys
from typing import Dict

import joblib
import numpy as np
import tensorflow as tf
from pysptools.abundance_maps.amaps import FCLS
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import InputSpec


def get_model(model_key: str, **kwargs):
    """
    Get a given instance of model specified by model_key.

    :param model_key: Specifies which model to use.
    :param kwargs: Any keyword arguments that the model accepts.
    """
    # Get the list of all model creating functions and their name as the key:
    all_ = {
        str(f): eval(f) for f in dir(sys.modules[__name__])
    }
    return all_[model_key](**kwargs)


def unmixing_linear(endmembers: np.ndarray, *args, **kwargs):
    class UnmixingLinear:
        def __init__(self, endmembers: np.ndarray):
            self.endmembers = endmembers

        def predict(self, samples: np.ndarray, *args, **kwargs):
            prediction = FCLS(M=samples, U=self.endmembers)
            return prediction.squeeze()

    return UnmixingLinear(endmembers)


from time import time


def unmixing_svr(dest_path: str, *args, **kwargs):
    class UnmixingSVR:
        def __init__(self, dest_path: str):
            self.svr = MultiOutputRegressor(SVR())
            self.dest_path = dest_path

        def fit(self, x: np.ndarray, y: np.ndarray, *args, **kwargs) -> Dict:
            start = time()
            self.svr = self.svr.fit(x, y)
            joblib.dump(self.svr,
                        os.path.join(dest_path, 'svr.pkl'), compress=9)
            return {'train_time': [time() - start]}

        def predict(self, x: np.ndarray, *args, **kwargs):
            return self.svr.predict(x)

        def compile(self, *args, **kwargs):
            pass

    return UnmixingSVR(dest_path)


def unmixing_pixel_based_cnn(n_classes: int, input_size: int,
                             **kwargs) -> tf.keras.Sequential:
    """
    Model for supervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv3D(filters=3, kernel_size=(1, 1, 5),
                               activation='relu',
                               input_shape=(1, 1, input_size, 1),
                               data_format='channels_last'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=6, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=12, kernel_size=(1, 1, 5),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=24, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=192, activation='relu'))
    model.add(tf.keras.layers.Dense(units=150, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def unmixing_cube_based_cnn(n_classes: int, input_size: int,
                            **kwargs) -> tf.keras.Sequential:
    """
    Model for supervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 1, 5),
                                     activation='relu',
                                     input_shape=(kwargs['neighborhood_size'],
                                                  kwargs['neighborhood_size'],
                                                  input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 5),
                                     activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Conv3D(filters=128, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=192, activation='relu'))
    model.add(tf.keras.layers.Dense(units=150, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def unmixing_pixel_based_dcae(n_classes: int, input_size: int,
                              **kwargs) -> tf.keras.Sequential:
    """
    Model for unsupervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=2, kernel_size=(1, 1, 3),
                                     activation='relu',
                                     input_shape=(1, 1, input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=4, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=8, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='relu'))
    model.add(tf.keras.layers.Softmax())
    # Decoder part (later to be dropped):
    model.add(tf.keras.layers.Dense(units=input_size, activation='relu'))
    # Set the endmembers weights to be equal to the endmembers matrix i.e.,
    # the spectral signatures of each class:
    model.layers[-1].set_weights(
        (np.swapaxes(kwargs['endmembers'], 1, 0), np.zeros(input_size)))
    # Freeze the last layer which must be equal to endmembers
    # and residual term (zero vector):
    model.layers[-1].trainable = False
    return model


def unmixing_cube_based_dcae(n_classes: int, input_size: int,
                             **kwargs) -> tf.keras.Sequential:
    """
    Model for unsupervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 3),
                                     activation='relu',
                                     input_shape=(kwargs['neighborhood_size'],
                                                  kwargs['neighborhood_size'],
                                                  input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=128, kernel_size=(1, 1, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='relu'))
    model.add(tf.keras.layers.Softmax())
    # Decoder part (later to be dropped):
    model.add(tf.keras.layers.Dense(units=input_size, activation='linear'))
    # Set the endmembers weights to be equal to the endmembers matrix i.e.,
    # the spectral signatures of each class:
    model.layers[-1].set_weights(
        (np.swapaxes(kwargs['endmembers'], 1, 0), np.zeros(input_size)))
    # Freeze the last layer which must be equal to endmembers
    # and residual term (zero vector):
    model.layers[-1].trainable = False
    return model


def get_index_of_named_layer(model, name: str) -> int:
    for idx, layer in enumerate(model.layers):
        if layer.name == name:
            return idx


def unmixing_deep_cnn(*args, **kwargs):
    model = tf.keras.Sequential()
    # 1
    model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(2, 2, 7),
                                     activation='relu',
                                     input_shape=(kwargs['neighborhood_size'],
                                                  kwargs['neighborhood_size'],
                                                  kwargs['input_size'], 1),
                                     data_format='channels_last',
                                     padding='same'))
    # 2
    model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(2, 2, 5),
                                     activation='relu', padding='same'))
    # 3
    model.add(tf.keras.layers.Conv3D(filters=128, kernel_size=(2, 2, 5),
                                     activation='relu', padding='same'))
    # 4
    model.add(tf.keras.layers.Conv3D(filters=256, kernel_size=(2, 2, 5),
                                     activation='relu', padding='same'))
    # 5
    model.add(tf.keras.layers.Conv3D(filters=512, kernel_size=(2, 2, 5),
                                     activation='relu'))
    # Dense
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=kwargs['n_classes'],
                                    activation='linear'))
    model.add(tf.keras.layers.Softmax())
    return model
