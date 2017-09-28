from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class MultiTower(object):
    def __init__(self, model, opt, num_gpus, tower_name, tb_logging=True):
        self.model = model
        self.opt = opt
        self.num_gpus = num_gpus
        self.tb_logging = tb_logging
        self.tower_name = tower_name
        
        self._loss = None

    def _loss_fn(self, logits, labels):
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
            name='cross_entropy_per_example'
        )
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')


    def _calc_tower_loss(self, scope, images, labels):
        """Calculate the total loss on a single tower running the CIFAR model.
        Args:
          scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
          images: Images. 4D tensor of shape [batch_size, height, width, 3].
          labels: Labels. 1D tensor of shape [batch_size].
        Returns:
           Tensor of shape [] containing the total loss for a batch of data
        """

        # Build inference Graph.
        logits = self.model.forward_pass(images)

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        _ = self._loss_fn(logits, labels)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses', scope)

        # Calculate the total loss for the current tower.
        total_loss = tf.add_n(losses, name='total_loss')

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            loss_name = re.sub('%s_[0-9]*/' % self.tower_name, '', l.op.name)
            tf.summary.scalar(loss_name, l)

        return total_loss

    def _calc_tower_grads(self, image, label):
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(self.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (self.tower_name, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        ###################################################################
                        #                    TOWER LOSS
                        ####################################################################
                        loss = self._calc_tower_loss(scope, image, label)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = self.opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        
        self._loss = loss
                        
        return tower_grads, summaries

    def calc_average_gradients(self, image_batch, label_batch):

        tower_grads, summaries = self._calc_tower_grads(
            image=image_batch, label=label_batch
        )

	average_grads = []
	for grad_and_vars in zip(*tower_grads):
	    # Note that each grad_and_vars looks like the following:
	    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
	    grads = []
	    for g, _ in grad_and_vars:
		# Add 0 dimension to the gradients to represent the tower.
		expanded_g = tf.expand_dims(g, 0)

		# Append on a 'tower' dimension which we will average over below.
		grads.append(expanded_g)

	    # Average over the 'tower' dimension.
	    grad = tf.concat(grads, 0)
	    grad = tf.reduce_mean(grad, 0)

	    # Keep in mind that the Variables are redundant because they are shared
	    # across towers. So .. we will just return the first tower's pointer to
	    # the Variable.
	    v = grad_and_vars[0][1]
	    grad_and_var = (grad, v)
	    average_grads.append(grad_and_var)

        if self.tb_logging:
            self._add_histograms_for_gradients(average_grads, summaries)

	return average_grads, summaries

    def _add_histograms_for_gradients(self, grads, summaries):
	for grad, var in grads:
	    if grad is not None:
		summaries.append(
		    tf.summary.histogram(var.op.name + '/gradients', grad)
		)
       
    def set_opt(self, new_opt):
        self.opt = new_opt

    def set_num_gpus(self, num_gpus):
        self.num_gpus = num_gpus
        
    def get_loss(self):
        return self._loss
