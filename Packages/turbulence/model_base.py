from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ModelBase(object):
    def __init__(self):
        pass
    
    def forward_pass(self, x):
        raise NotImplementedError(
            "forward_pass() is implemented in Model sub classes.")

    def _dense(self, x, units):
        return tf.layers.dense(
            inputs=x, units=units, activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
       

    def _conv(self, x, filters, kernel_size=3, strides=1):
        return tf.layers.conv2d( 
            inputs=x, filters=filters, kernel_size=kernel_size,
            strides=strides, padding='SAME', data_format='channels_first',
            activation=None, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )


    #################################################################
    #                     POOLING
    #################################################################
    def _max_pool(self, x, pool_size=2, strides=2):
        return tf.layers.max_pooling2d(
            inputs=x, pool_size=pool_size, strides=strides,
            data_format="channels_first"
        )

    def _avg_pool(self, x, pool_size=2, strides=2):
        output_tensor = tf.layers.average_pooling2d(
            inputs=x, pool_size=pool_size, strides=strides,
            padding="VALID", data_format="channels_first"
        )
        return output_tensor

    def _gap(self, x):
        shape = x.get_shape().as_list()
        height, width = shape[2:]
        pool_size = (height, width)
        u = self._avg_pool(x, pool_size=pool_size, strides=1)
        output_tensor = tf.squeeze(u, axis=[2, 3])
        return output_tensor 


    #################################################################
    #                     REGULARIZATION
    #################################################################
    def _dropout(self, x, rate, is_training):
        return tf.layers.dropout(x, rate=rate, training=is_training)

    def _bn(self, x, is_training, axis=-1):
        return tf.layers.batch_normalization(
            inputs=x, axis=axis, training=is_training
        )

    ###############################################################
    #                 ACTIVATION FUNCTION
    ###############################################################
    def _relu(self, x):
        return tf.nn.relu(x)

    def _selu(self, x):
        return tf.nn.selu(x)

    def _maxout(self, x, units, extractors, name="MaxOut"):
        '''
        ref. Ian J. Goodfellow et al. Maxout Networks.
        ref. Qi Wang, Joseph JaJa. From Maxout to Channel-Out: Encoding Information on Sparse Pathways.
        ref. http://www.simon-hohberg.de/2015/07/19/maxout.html
        m: Number of units in each linear feature extractor (complexity)
        k: Number of linear feature extractors
        '''
        d = x.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W', shape=[d, units, extractors],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.get_variable(
                name='b', shape=[units, extractors],
                initializer=tf.zeros_initializer()
            )
            # shap(z) = [N, m, k]
            z = tf.tensordot(x, W, axes=[[1], [0]]) + b
            # shape(h) = [N, m]
            h = tf.reduce_max(z, axis=-1)
        return h

