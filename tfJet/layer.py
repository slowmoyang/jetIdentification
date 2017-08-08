import tensorflow as tf
from layer_utils import create_var
from layer_utils import flatten


def maxout(x, m, k, name='maxout'):
    '''
    ref. Ian J. Goodfellow et al. Maxout Networks.
    ref. Qi Wang, Joseph JaJa. From Maxout to Channel-Out: Encoding Information on Sparse Pathways.
    ref. http://www.simon-hohberg.de/2015/07/19/maxout.html

    m: Number of units in each linear feature extractor (complexity)
    k: Number of linear feature extractors
    '''
    d = x.get_shape().as_list()[-1]
    
    with tf.variable_scope(name):
        W = tf.get_variable('W', shape=[d, m, k], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[m, k], initializer=tf.contrib.layers.xavier_initializer())

        # x: [N, d]
        # W: [d, m, k]
        # z: [N, m, k]
        z = tf.tensordot(x, W, axes=[[1], [0]]) + b

        h = tf.reduce_max(z, axis=-1) # shape: [N, m]
    
    return h    


def fc_layer(input_tensor, output_dim, layer_name, DW=False, act_ftn=tf.nn.relu):
    input_dim = input_tensor.get_shape().as_list()[-1]

    with tf.variable_scope(layer_name):
        W = create_var(var_type='weight', input_dim=input_dim, output_dim=output_dim, var_name='weight', DW=DW)
        b = create_var(var_type='bias', output_dim=output_dim, var_name='bias')
        preactivate = tf.nn.bias_add(tf.matmul(input_tensor, W), b)
        output_tensor = act_ftn(preactivate, name='activation')

    return output_tensor


def conv_layer(input_tensor, output_channels, layer_name,
               stride=1,kernel_size=3, DW=False, VK=True):
    input_channels = input_tensor.get_shape().as_list()[1]

    with tf.variable_scope(layer_name):
        kernel = create_var(var_type='kernel', input_channels=input_channels, output_channels=output_channels, kernel_size=kernel_size, var_name='kernel', DW=DW)
        strides = [1, 1, stride, stride]
        preactivate = tf.nn.conv2d(input = input_tensor, filter = kernel, strides = strides, padding = 'SAME', data_format='NCHW')
        output_tensor = tf.nn.relu(preactivate)

    return output_tensor


def bn_layer(input_tensor, output_dim, name, bn_epsilon=0.001, data_format='NCHW'):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_tensor, axes=[0, 2, 3])
        beta = tf.get_variable('beta', output_dim, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', output_dim, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, bn_epsilon)

    return bn_layer

 
def max_pooling_layer(input_tensor, layer_name, k=2, s=2):
    """

    """
    ksize = [1, 1, k, k]
    strides = [1, 1, s, s]

    with tf.name_scope(layer_name):
        output_tensor = tf.nn.max_pool(input_tensor, ksize=ksize , strides=strides, padding='SAME', data_format='NCHW')
    return output_tensor

def gap_layer(input_tensor, layer_name):
    """
    global average pooling
    ref. Min Lin, Qiang Chen, Shuicheng Yan. Network In Network. arXiv:1312.4400
    """
    shape = input_tensor.get_shape().as_list()
    H, W = shape[2:]
    ksize = strides = [1, 1, H, W]

    with tf.variable_scope(layer_name):
        unflattened = tf.nn.avg_pool(input_tensor, ksize=ksize, strides=strides, padding='SAME', data_format='NCHW')
        output_tensor = flatten(unflattened)

    return output_tensor

def maxout(input_tensor, layer_name):
    input_dim = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope():
        pass
