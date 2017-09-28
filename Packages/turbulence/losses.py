from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import math_ops

def calc_l2_weight_decay(weight_decay_rate):
    '''ref. https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py ''' 
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find('DW') > 0:
            costs.append(tf.nn.l2_loss(var))
    return tf.multiply(weight_decay_rate, tf.add_n(costs))


def calc_xentropy(logits, labels, DW=False):
    with tf.name_scope("loss"):
        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            cross_entropy
            loss = cross_entropy
        if DW:
            with tf.name_scope("l2_weight_decay"):
                weight_decay = calc_l2_weight_decay(weight_decay_rate=0.01)
            loss += weight_decay
    return loss

def calc_central_loss():
    pass


def _weak_loss_v1_(weak_labels, predictions, gluon_idx):
    with tf.name_scope("weak_loss"):
        gluon_probs_mean = tf.reduce_mean(preds[:, gluon_idx])
        weak_labels_mean = tf.reduce_mean(weak_labels)

        weak_loss = tf.abs(gluon_probs_mean - weak_labels_mean)
    tf.summary.scalar("weak_loss", weak_loss)
    return weak_loss

def _weak_loss_v2(weak_labels, predictions, gluon_idx):
    with tf.name_scope("weak_loss"):
        gluon_prob = tf.expand_dims(
            input=predictions[:, gluon_idx],
            axis=-1
        )
        abs_diff = tf.abs(gluon_prob - weak_labels)
        weak_loss = tf.reduce_mean(abs_diff)
    return weak_loss

def calc_weak_loss(weak_labels, predictions, gluon_idx=1, ver=2):
    if ver==1:
        return _weak_loss_v1(weak_labels, predictions, gluon_idx)
    elif ver==2:
        return _weak_loss_v2(weak_labels, predictions, gluon_idx)
    else:
        NotImplementedError("")

