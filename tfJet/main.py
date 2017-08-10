from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf

from training import train
from evaluation import evaluate
from qg_histogram import draw_all_qg_histograms
from utils import get_log_dir

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'training_data',
#    '../data/jet_pT-ALL_eta-below_2.4_pythia/tfrecords/jet_training_7896.tfrecords',
    '../data/jet_pT-ALL_eta-below_2.4_pythia/tfrecords/jet_training_1111518.tfrecords',
    'the training data set'
)
tf.app.flags.DEFINE_string(
    'validation_data',
#    '../data/jet_pT-ALL_eta-below_2.4_pythia/tfrecords/jet_validation_2632.tfrecords',
    '../data/jet_pT-ALL_eta-below_2.4_pythia/tfrecords/jet_validation_370506.tfrecords',
    'the validation data set'
)


tf.app.flags.DEFINE_integer('batch_size', 1000, 'batch size')
tf.app.flags.DEFINE_integer('num_epochs', 30, 'the number of epochs')
tf.app.flags.DEFINE_float('initial_lr', 0.001, 'ininital learning rate')
tf.app.flags.DEFINE_float('dropout_prob', 0.5, 'the probability of dropout')

log_dir = get_log_dir(dname='maxout_test', creation=True)

log_file_path = os.path.join(log_dir.path, 'log.txt')
log = open(log_file_path, 'w')

# ENVIRONMENT
log.write('Test Date: %s' % time.asctime())
log.write('Training Data: %s\n' % FLAGS.training_data)
log.write('Validation Data: %s\n' % FLAGS.validaiton_data)
log.write('model: %s\n' % 'temp')
log.write('optimizer: %s\n' % 'Adam')
# CONSTANT
log.write('Batch Size: %d\n' % FLAGS.batch_size)
log.write('#Epochs: %d\n' % FLAGS.num_epochs)
# HYPERPARAMETER
log.write('Initial Learning Rate: %.4f\n' % FLAGS.initial_lr)
log.write('Probabilty of Dropout: %.3f\n' % FLAGS.dropout_prob)

tr_start_time = time.time()
train(
    tfrecords_path=FLAGS.training_data,
    tfevents_dir=log_dir.tfevents.path,
    ckpt_dir=log_dir.ckpt.path,
    benchmark_path=log_dir.path,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs,
    initial_lr=FLAGS.initial_lr,
    dropout_prob=FLAGS.dropout_prob,
)
duration = time.time() - tr_start_time
duration_h = duration / 360
log.write('Training Time: %.2f\n' % duration_h)

evaluate(
    training_data=FLAGS.training_data,
    validation_data=FLAGS.validation_data,
    log_dir=log_dir
)

draw_all_qg_histograms(qg_histogram_dir=log_dir.qg_histogram)


log.close()