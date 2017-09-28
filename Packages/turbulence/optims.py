from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def training(loss, lr, optimizer=tf.train.AdamOptimizer):
    with tf.name_scope("train"):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer(learning_rate=lr).minimize(loss, global_step=global_step)
    return train_op


def get_decayed_lr(global_step,
                   num_examples_per_epoch_for_train,
                   batch_size,
                   num_epochs_per_decay,
                   initial_learning_rate,
                   learning_rate_decay_factor):
    # Calculate the learning rate schedule.
    num_batches_per_epoch = (num_examples_per_epoch_for_train / batch_size)
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=learning_rate_decay_factor,
        staircase=True
    )
    return lr
