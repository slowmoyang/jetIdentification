from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 

import time

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from drawer import get_date


def loss(logits, labels):
    with tf.name_scope("cross_entropy"):
        diff = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy

def l2_weight_decay(weight_decay_rate):
    '''ref. https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py ''' 
    costs = []
    for var in tf.trainable_variable():
        if var.op.name.find(r'DW') > 0:
            costs.append(tf.nn.l2_loss(var))
    return tf.multiply(weight_decay_rate, tf.add_n(costs))


def training(loss, lr):
    with tf.name_scope("train"):
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    return train_op

def evaluation(logits, labels):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        with tf.name_scope('num_correct'):
            num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy
