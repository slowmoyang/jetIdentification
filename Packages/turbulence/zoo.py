from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layer import fc_layer
from layer import conv_layer
from layer import max_pooling_layer
from layer import gap_layer
from layer import gmp_layer
from layer import maxout
from layer_utils import flatten
import module


def vgg8(images, keep_prob):
    conv1 = conv_layer(images, 64, 'conv1')
    conv2 = conv_layer(conv1, 64, 'conv2')
    pool1 = max_pooling_layer(conv2, layer_name='max_pool_1') 

    conv3 = conv_layer(pool1, 128, 'conv3')
    conv4 = conv_layer(conv3, 128, 'conv4')
    pool2 = max_pooling_layer(conv4, layer_name='max_pool_2') 

    conv5 = conv_layer(pool2, 256, 'conv5')
    conv6 = conv_layer(conv5, 256, 'conv6')
    pool3 = max_pooling_layer(conv6, layer_name='max_pool_3')

    conv7 = conv_layer(pool3, 512, 'conv7')
    conv8 = conv_layer(conv7, 512, 'conv8')
    pool4 = max_pooling_layer(conv8, layer_name='max_pool_4') 

    flat = flatten(pool4)

    fc1 = fc_layer(input_tensor = flat, output_dim = 512, layer_name = 'fc1')
    fc2 = fc_layer(input_tensor = fc1, output_dim = 256, layer_name = 'fc2')
    fc3 = fc_layer(input_tensor = fc2, output_dim = 128, layer_name = 'fc3')

    with tf.name_scope('dropout'):
        dropped = tf.nn.dropout(fc3, keep_prob)

    logits = fc_layer(input_tensor=dropped, output_dim=2, layer_name='logits', act_ftn=tf.identity)
    return logits


def vgg8_gap(images, keep_prob):
    # images [N, 3, 33, 33]
    conv1 = conv_layer(images, 64, 'conv1')
    conv2 = conv_layer(conv1, 64, 'conv2')
    pool1 = max_pooling_layer(conv2, layer_name='max_pool_1') 

    conv3 = conv_layer(pool1, 128, 'conv3')
    conv4 = conv_layer(conv3, 128, 'conv4')
    pool2 = max_pooling_layer(conv4, layer_name='max_pool_2') 

    conv5 = conv_layer(pool2, 256, 'conv5')
    conv6 = conv_layer(conv5, 256, 'conv6')
    pool3 = max_pooling_layer(conv6, layer_name='max_pool_3')

    conv7 = conv_layer(pool3, 512, 'conv7')
    conv8 = conv_layer(conv7, 512, 'conv8')
    
    # Global average pooling
    gap = gap_layer(conv8, layer_name='global_avg_pool')

    fc = fc_layer(gap, 128, 'fc')

    with tf.name_scope('dropout'):
        dropped = tf.nn.dropout(fc, keep_prob)

    logits = fc_layer(input_tensor=dropped, output_dim=2, layer_name='logits', act_ftn=tf.identity)
    return logits


def maxout_network(images, keep_prob):
    # images [N, 3, 33, 33]
    conv1 = conv_layer(images, 64, 'conv1')
    conv2 = conv_layer(conv1, 64, 'conv2')
    pool1 = max_pooling_layer(conv2, layer_name='max_pool_1') 

    conv3 = conv_layer(pool1, 128, 'conv3')
    conv4 = conv_layer(conv3, 128, 'conv4')
    pool2 = max_pooling_layer(conv4, layer_name='max_pool_2') 

    conv5 = conv_layer(pool2, 256, 'conv5')
    conv6 = conv_layer(conv5, 256, 'conv6')
    pool3 = max_pooling_layer(conv6, layer_name='max_pool_3')

    conv7 = conv_layer(pool3, 512, 'conv7')
    conv8 = conv_layer(conv7, 512, 'conv8')
    
    # Global average pooling
    gap = gap_layer(conv8, layer_name='global_avg_pool')

    mo = maxout(x=gap, m=128, k=10, name='maxout')

    with tf.name_scope('dropout'):
        dropped = tf.nn.dropout(mo, keep_prob)

    logits = maxout(x=dropped, m=2, k=10, name='logits')

    return logits

def mon2(images, keep_prob):
    # images [N, 3, 33, 33]
    conv1 = conv_layer(images, 64, 'conv1')
    conv2 = conv_layer(conv1, 64, 'conv2')
    pool1 = max_pooling_layer(conv2, layer_name='max_pool_1') 

    conv3 = conv_layer(pool1, 128, 'conv3')
    conv4 = conv_layer(conv3, 128, 'conv4')
    pool2 = max_pooling_layer(conv4, layer_name='max_pool_2') 

    conv5 = conv_layer(pool2, 256, 'conv5')
    conv6 = conv_layer(conv5, 256, 'conv6')
    pool3 = max_pooling_layer(conv6, layer_name='max_pool_3')

    conv7 = conv_layer(pool3, 512, 'conv7')
    conv8 = conv_layer(conv7, 512, 'conv8')
    
    # Global average pooling
    gap = gap_layer(conv8, layer_name='global_avg_pool')

    mo = maxout(x=gap, m=128, k=5, name='maxout')

    with tf.name_scope('dropout'):
        dropped1 = tf.nn.dropout(mo, keep_prob)

    logits = maxout(x=dropped1, m=2, k=5, name='logits')

    with tf.name_scope('dropout'):
        dropped2 = tf.nn.dropout(logits, keep_prob)


    return dropped2




def custom170809(image, keep_prob, is_training):
    # image [B, 3, 33, 33]

    # 1st feature map: [B, 32, 17, 17]
    fm1 = conv_layer(input_tensor=image, output_channels=32, layer_name='conv')
    # 2nd feature map: [B, 64, 9, 9]
    fm2 = module.full_preact(fm1, is_training=is_training, module_name='fpa1')
    # 3rd feature map: [B, 128, 5, 5]
    fm3 = module.full_preact(fm2, is_training=is_training, module_name='fpa2')
    # 4th feature map: [B, 256, 3, 3]
    fm4 = module.full_preact(fm3, is_training=is_training, module_name='fpa3')
    # global average/max pooling layer: [B, 256]
    gap = gap_layer(fm4, 'gap')
    gmp = gmp_layer(fm4, 'gmp')
    # concat: [B, 512]
    concat = tf.concat([gap, gmp], axis=1)
    # maxout: [B, 64]
    mo = maxout(x=concat, m=64, k=5, name='maxout')
    with tf.name_scope('dropout'):
        dropped = tf.nn.dropout(mo, keep_prob=keep_prob)
    # logits: [B, 2]
    logits = fc_layer(input_tensor=dropped, output_dim=2, layer_name='logits', act_ftn=tf.identity)
    return logits
