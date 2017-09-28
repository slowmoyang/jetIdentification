from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model_base import ModelBase

class DenseNet(ModelBase):
    '''
    DenseNet-B : Use Bottleneck layers as a non-linear transformation.
    DenseNet-C : the DenseNet with compression_factor < 1
    DenseNet-BC : 
    '''
    def __init__(self, num_l, is_training):
        super(DenseNet, self).__init__()
        self._is_training = is_training

    def forward_pass(self, num_block):
        # Before entering the first dense block, a convolution
        # with 16 (or twice the growth rate for DenseNet-BC)
        # output channels is performed on the input images.
        x = self._conv(x, filters=filters, kernel_size, strides)

        for i in range(num_block-1):
            x = self._dense_block(
            x = self._transition(

        x = self._dense_block(

        # At the end of the last dense block, a global
        # average pooling is performed and then a softmax
        # classifier is attached.
        x = self._bn
        x = tf.nn.relu
        x = self._gap(x)

        

    def _composite_fn(self, x, filters, name):
        with tf.name_scope(name):
            x = self._bn(x, self._is_training, axis=1)
            x = tf.nn.relu(x)
            x = self._conv(x, filters=filters, kernel_size=3, strides=1)
        return x

    def _dense_block(self, x, num_layers, growth_rate=12):
        for i in range(num_layers):
            x_i = self._composite_fn(x, filters=growth_rate)
            x = tf.concat(axis=1, [x, x_i])
        return x

    def _transition(self, x, compression_factor):
        in_filters = int(x.get_shape()[1])
        out_filters = int(in_filters * compression_facotr)
        with tf.name_scope(scope):
            x = self._bn(x, self._is_training, axis=1)
            x = self._conv(x, filters=out_filters, kernel_size=1, strides=1)
            x = self._avg_pool(x, pool_size=2, strides=2)
        return x

    def _bottleneck(self, x, name):
        with tf.name_scope(name):
            x = self._bn(x, self._is_training, axis=1)
            x = tf.nn.relu(x)             
            x = self._conv(x, filters=, kernel_size=1, strides=1)
            x = self._bn(x, self._is_training, axis=1)
            x = tf.nn.relu(x)
            x = self._conv(x, filters=, kernel_size=3, strides)
        return x
