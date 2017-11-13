from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import (
    Activation,
    Conv2D,
    MaxPooling2D,
    BatchNormalization as BN,
    GlobalAveragePooling2D as GAP,
    Dense,
    Dropout
)

from keras.engine.topology import Layer

from keras.utils import multi_gpu_model
from keras import backend as K

def build_a_cnn_stem(input_shape=(1, 99, 99)):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='elu', input_shape=input_shape))
    model.add(BN(axis=1))
    model.add(Conv2D(32, (3,3), activation='elu', padding="SAME"))
    model.add(BN(axis=1))
    model.add(Conv2D(32, (3,3), activation='elu', padding="SAME"))
    model.add(MaxPooling2D((2,2)))
    model.add(BN(axis=1))
    model.add(Conv2D(64, (3,3), activation='elu', padding="SAME"))
    model.add(BN(axis=1))
    model.add(Conv2D(64, (3,3), activation='elu', padding="SAME"))
    model.add(BN(axis=1))
    model.add(Conv2D(64, (3,3), activation='elu', padding="SAME"))
    model.add(BN(axis=1))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(MaxPooling2D((2,2)))
    model.add(BN(axis=1))
    model.add(Conv2D(128, (3,3), activation='elu', padding="SAME"))
    model.add(BN(axis=1))
    model.add(Conv2D(128, (3,3), activation='elu', padding="SAME"))
    model.add(BN(axis=1))
    model.add(Conv2D(128, (3,3), activation='elu', padding="SAME"))
    model.add(MaxPooling2D((2,2)))
    return model


def _conv_gap(stem, num_classes):
    """
    """
    model = Sequential()
    model.add(stem)

    model.add(
        Conv2D(filters=num_classes, kernel_size=(1, 1))
    )
    model.add(GAP())

    if num_classes == 1:
        act_fn = "sigmoid"
    elif num_classes > 1:
        act_fn = "softmax"
    else:
        raise ValueError("")

    # model.add(Activation(act_fn))
    return model


def add_an_output_layer(stem, num_classes=1):
    return _conv_gap(stem, num_classes)

def add_an_sigmoid_layer(model):
    sig = Sequential()
    sig.add(model)
    sig.add(Activation("sigmoid"))
    return sig
