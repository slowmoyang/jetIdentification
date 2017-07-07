import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import xavier_initializer_conv2d

def if_data_format(data_format, case_nchw, case_nhwc):
    if data_format == 'NCHW':
        return case_nchw
    elif data_format == 'NHWC':
        return case_nhwc
    else:
        raise ValueError("data_format has to be either 'NCHW' or 'NHWC'")

def get_xavier(shape, is_conv2d=True, var_name='xaiver_weights'):
    # get_variable(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True,
    #              collections=None, caching_device=None, partitioner=None, validate_shape=True, custom_getter=None)
    if is_conv2d:
        W = tf.get_variable(var_name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    else:
        W = tf.get_variable(var_name, shape, initializer=tf.contrib.layers.xavier_initializer())
    return W

def get_weight(shape, layer_name, initializer=xavier_initializer):
    d = len(shape)
    if d == 2:
        name = 'weight_in_'+layer_name
    elif d == 1:
        name = 'biases_in_'+layer_name
    return tf.get_variable(name, shape, initializer=initializer())

def get_filter(input_channels, output_channels, layer_name, data_format='NCHW',
               kernel_size = 3, initializer=xavier_initializer_conv2d):
    '''
      NCHW format
    '''
    name = 'filters_in_' + layer_name
    shape = [kernel_size, kernel_size, input_channels, output_channels]
    return tf.get_variable(name, shape, initializer=initializer())

def flatten(input_tensor):
    input_shape = input_tensor.get_shape().as_list()
    output_dim = reduce(lambda x, y: x*y, input_shape[1:])
    output_tensor = tf.reshape(tensor=input_tensor, shape=(-1, output_dim))
    return output_tensor

def fc_layer(input_tensor, output_dim, layer_name, act_ftn=tf.nn.relu):
    input_dim = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(layer_name):
        W = get_weight([input_dim, output_dim], layer_name)
        b = get_weight([output_dim], layer_name)
        preactivate = tf.matmul(input_tensor, W) + b
        activations = act_ftn(preactivate, name='activation')
    return activations

def conv_layer(input_tensor, output_channels, layer_name,
               stride=1,kernel_size=3, data_format='NCHW'):
    input_channels = input_tensor.get_shape().as_list()[1]
    with tf.variable_scope(layer_name):
        filter = get_filter(input_channels, output_channels, layer_name, kernel_size)
        strides = if_data_format(data_format, case_nchw=[1, 1, stride, stride],
                                              case_nhwc=[1, stride, stride, 1])
        preact = tf.nn.conv2d(input = input_tensor, filter = filter, strides = strides, padding = 'SAME', data_format=data_format)
        output_tensor = tf.nn.relu(preact)
    return output_tensor



def bn_layer(input_tensor, output_dim, name, bn_epsilon=0.001, data_format='NCHW'):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    moment_axes = if_data_format(data_format, case_nchw=[0, 2, 3], case_nhwc=[0, 1, 2])

    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_tensor, axes=moments_axes)
        beta = tf.get_variable('beta', output_dim, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', output_dim, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, bn_epsilon)
    return bn_layer
 
def max_pooling_layer(input_tensor, layer_name, data_format='NCHW', k=2, s=2):
    """

    """
    ksize = if_data_format(data_format, case_nchw=[1, 1, k, k], case_nhwc=[1, k, k, 1])
    strides = if_data_format(data_format, case_nchw=[1, 1, s, s], case_nhwc=[1, s, s, 1])


    with tf.name_scope(layer_name):
        output_tensor = tf.nn.max_pool(input_tensor, ksize=ksize , strides=strides, padding='SAME', data_format=data_format)
    return output_tensor

def gap_layer(input_tensor, layer_name, data_format='NCHW'):
    """
    global average pooling
    ref. Min Lin, Qiang Chen, Shuicheng Yan. Network In Network. arXiv:1312.4400
    """
    shape = input_tensor.get_shape().as_list()
    if data_format == 'NCHW':
        H, W = shape[2:]
        ksize = strides = [1, 1, H, W]
    elif data_format == 'NHWC':
        H, W = shape[1:3]
        ksize = strides = [1, H, W, 1]
    else:
        raise ValueError("Ewww..")
    with tf.variable_scope(layer_name):
        unflattened = tf.nn.avg_pool(input_tensor, ksize=ksize, strides=strides, padding='SAME', data_format=data_format)
        output_tensor = flatten(unflattened)
    return output_tensor


def vgg_module(input_tensor, num_block, module_name):
    '''
      full pre-activation
      NCHW format

      cin = the channels of the input tensor
      cout = the channels of the output tensor

      tin = the input tensor
      tout = the output tensor
    '''
    input_channels = input_tensor.get_shape().as_list()[1]
    output_channels = 2 * input_channels
    output_tensor = input_tensor
    with tf.variable_scope(module_name):
        for i in range(num_block):
            # batch normalization
            bn_name = 'bn-%d_in_%s' %(i, module_name)
            output_tensor = tf.contrib.layers.batch_norm(output_tensor, fused=True, data_format='NCHW')
            print(bn_name, output_tensor.get_shape())
            # activation
            output_tensor = tf.nn.relu(output_tensor)
            # convolution
            conv_name = 'conv-%d_in_%s' % (i, module_name)
            output_tensor = conv_layer(output_tensor, output_channels,layer_name=conv_name)
            print(conv_name, output_tensor.get_shape())
        # pooling layer
        output_tensor = tf.nn.max_pool(output_tensor, ksize=[1, 1, 2, 2] , strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
        print('max_pooling in ', module_name, output_tensor.get_shape())
    return output_tensor

def resnet_module(input_tensor):
    pass

def inception_module(input_tensor):
    pass

