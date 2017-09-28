import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

from layer import *

# full pre-activation
def full_preact(input_tensor, is_training, module_name, num_block=2,
                               DW=False, VK=True):
    input_channels = input_tensor.get_shape().as_list()[1]
    output_channels = 2 * input_channels

    output_tensor = input_tensor

    with tf.variable_scope(module_name):
        for i in range(num_block):
            # batch normalization & ReLU
            output_tensor = batch_norm(
                inputs=output_tensor,
                decay=0.9,
                activation_fn=tf.nn.relu,
                is_training=is_training
            )

            # convolution
            output_tensor = conv_layer(
                input_tensor=output_tensor,
                output_channels=output_channels,
                layer_name='conv_%d' % i,
                DW=DW,
                VK=VK
            )

        # pooling layer
        output_tensor = tf.nn.max_pool(
            output_tensor,
            ksize=[1, 1, 2, 2] ,
            strides=[1, 1, 2, 2],
            padding='SAME',
            data_format='NCHW'
        )

    return output_tensor





def resnet_module(input_tensor):
    pass


# inception module, naive version
def inception_naive(input_tensor, channels, module_name):
    with tf.variable_scope(module_name):
        conv1x1 = conv_layer(input_tensor=input_tensor, kernel_size=1, output_channels=channels['conv1x1'], layer_name='conv1x1')
        conv3x3 = conv_layer(input_tensor=input_tensor, kernel_size=3, output_channels=channels['conv3x3'], layer_name='conv3x3')
        conv5x5 = conv_layer(input_tensor=input_tensor, kernel_size=5, output_channels=channels['conv5x5'], layer_name='conv5x5')
        pool3x3 = max_pooling_layer(input_tensor, layer_name='max_pool3x3', k=3, s=1)

        # axis: channels
        output_tensor = tf.concat(values=[conv1x1, conv3x3, conv5x5, pool3x3], axis=1)

    return output_tensor


# Inception module with dimension reductions
def inception_v1(input_tensor, channels, module_name):
    with tf.variable_scope(module_name):
        with tf.variable_scope('branch1x1'):
            branch1x1 = conv_layer(input_tensor=input_tensor, kernel_size=1, output_channels=channels['#1x1'], layer_name='conv1x1')
        with tf.variable_scope('branch3x3'):
            branch3x3 = conv_layer(input_tensor=input_tensor, kernel_size=1, output_channels=channels['#3x3_reduce'], layer_name='conv1x1')
            branch3x3 = conv_layer(input_tensor=branch3x3, kernel_size=3, output_channels=channels['#3x3'], layer_name='conv3x3')
        with tf.variable_scope('branch5x5'):
            branch5x5 = conv_layer(input_tensor=input_tensor, kernel_size=1, output_channels=channels['#5x5_reduce'], layer_name='conv1x1')
            branch5x5 = conv_layer(input_tensor=branch5x5, kernel_size=5, output_channels=channels['#5x5'], layer_name='conv5x5')
        with tf.variable_scope('branch1x1'):
            branch_pool = max_pooling_layer(input_tensor, layer_name='max_pool3x3', k=3, s=1)
        output_tensor = tf.concat([branch1x1, branch3x3, branch5x5, branch_pool], axis=1)
    return output_tensor


# factorizing inception module
def inception_module_v2(input_tensor, channels, module_name):
    pass

def dense_net_module():
    pass
