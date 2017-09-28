from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import re
import time
import numpy as np
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from mnist_model import CustomModel
from mnist_pipeline import get_dataset
from mnist_optims import get_decayed_lr
from gpus_utils import MultiTower

sys.path.append("../turbulence")
from utils import count_num_examples 


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

DATA_DIR = "../data"
TRAINING_FILE = '../data/train.tfrecords'
VALIDATION_FILE = '../data/validation.tfrecords'
TRAIN_DIR='../logs'
TOWER_NAME = 'tower'
IMAGE_PIXELS = 784

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

BATCH_SIZE = 64
NUM_GPUS = 2
NUM_EPOCHS = 10


tf.logging.set_verbosity(tf.logging.FATAL)


def train(tb_logging=True):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals
        # the number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            name='global_step',
            shape=[],
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        summaries = []

        ########################################################
        #                   INPUT PIPELINE
        #######################################################
        training_dataset = get_dataset(
            filenames=TRAINING_FILE,
            batch_size=64
        )
        validation_dataset = get_dataset(
            filenames=VALIDATION_FILE,
            batch_size=500
        )

        handle = tf.placeholder(tf.string, shape=[])

        iterator = tf.contrib.data.Iterator.from_string_handle(
            string_handle=handle,
            output_types=training_dataset.output_types,
            output_shapes=training_dataset.output_shapes
        )

        batch = iterator.get_next()

        training_iterator = training_dataset.make_initializable_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()

        ########################################################
        #                       MODEL
        ########################################################

        is_training = tf.placeholder(dtype=tf.bool)

        units_list = [10, 25, 50, 100, 50, 25, 10, 2],
        seq=["Dense", "ReLU", "BN"],

        model = CustomModel(
            units_list=unist_list,
            seq=seq,
            is_training=is_training
        )

        logits = model.forward_pass(batch['variables'])

        predictions = tf.nn.softmax(logits)

        #########################################################
        #                 OPTIMIZATION ALGORITHM
        #########################################################

        num_examples_per_epoch_for_train = count_num_examples(
            filenames=training_files
        )

        lr = get_decayed_lr(
            global_step=global_step,
            num_examples_per_epoch_for_train=num_examples_per_epoch_for_train,
            batch_size=BATCH_SIZE,
            num_epochs_per_decay=NUM_EPOCHS_PER_DECAY,
            initial_learning_rate=INITIAL_LEARNING_RATE,
            learning_rate_decay_factor=LEARNING_RATE_DECAY_FACTOR
        )


        optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr,
            momentum=0.9,
            use_nesterov=True,
            use_locking=True
        )

        ###########################################################
        #                       METRICS
        ###########################################################

        accuracy = calc_accuracy(logits=logits, labels=batch['label'])


        ###########################################################
        #                     MULTIPLE TOWERS
        ###########################################################
        multi_tower = MultiTower(
            model=model, opt=optimizer,
            num_gpus=2, tower_name=TOWER_NAME
        )

        grads, grads_summaries = multi_tower.calc_average_gradients(
            image_batch=batch['image'], label_batch=batch['label']
        )

        if tb_logging:
            summaries += grads_summaries
            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', lr))

        ###########################################################
        #                  TRAINING OPERATION
        ###########################################################

        train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        if tb_logging:
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))


        #############################################################
        #
        #############################################################

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(),sharded=True)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        # init = tf.global_variables_initializer()

        # The op for initializing the variables.
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True
            )
        )

        sess.run(init_op)

        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter("../logs", sess.graph)

        ##################################################################
        #                    TRAINING LOOP WITH VALIDATION
        ##################################################################
        step = 0
        for epoch in range(NUM_EPOCHS):
            sess.run(training_iterator.initializer)
            sess.run(validation_iterator.initializer)
            while True:
                try:
		    start_time = time.time()

		    # Run one step of the model.  The return values are
		    # the activations from the `train_op` (which is
		    # discarded) and the `loss` op.  To inspect the values
		    # of your ops or variables, you may include them in
		    # the list passed to sess.run() and the value tensors
		    # will be returned in the tuple from the call.
		    training_fetches = [train_op]
		    training_feed = {handle: training_handle} 
		    _ = sess.run(
			fetches=training_fetches,
			feed_dict=training_feed
		    )

		    duration = time.time() - start_time

		    # assert not np.isnan(
		    #    loss_value), 'Model diverged with loss = NaN'

		    # Print an overview fairly often.
		    if step % 100 == 0:
			num_examples_per_step = BATCH_SIZE * NUM_GPUS
			examples_per_sec = num_examples_per_step / duration
			sec_per_batch = duration / NUM_GPUS
			format_str = (
			    '%s: step %d,  (%.1f examples/sec; %.3f '
			    'sec/batch)')
			print(format_str % (datetime.now(), step,
					    examples_per_sec, sec_per_batch))
		    if True:
			if step % 10 == 0:
			    validation_feed = {handle: validation_handle}
			    summary_str = sess.run(summary_op, feed_dict=validation_feed)
			    summary_writer.add_summary(summary_str, step)

		    # Save the model checkpoint periodically.
		    if step % 1000 == 0 or (
			step + 1) == NUM_EPOCHS * BATCH_SIZE:
			checkpoint_path = os.path.join(TRAIN_DIR,
						       'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=step)

		    step += 1
                except tf.errors.OutOfRangeError:
                    break

        # Wait for threads to finish.
        coord.request_stop()
        coord.join(threads)
        sess.close()

def evaluate():
    """Eval MNIST for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels for MNIST.
        mnist = input_data.read_data_sets(DATA_DIR, one_hot=False)
        images = mnist.test.images
        labels = mnist.test.labels

        # Build a Graph that computes the logits predictions from the
        # inference model.
        model = CustomModel(tower_name=TOWER_NAME)
        logits = model.forward_pass(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)

        # Create saver to restore the learned variables for eval.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            predictions = np.sum(sess.run([top_k_op]))

            # Compute precision.
            print('%s: precision = %.3f' % (datetime.now(), predictions))


def main(argv=None):  # pylint: disable=unused-argument
    start_time = time.time()
    train()
    duration = time.time() - start_time
    print('Total Duration (%.3f sec)' % duration)
    evaluate()

if __name__ == '__main__':
    tf.app.run()
