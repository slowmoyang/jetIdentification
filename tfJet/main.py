from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from train import train
from evaluation import evaluate
from qg_histogram import draw_all_qg_histograms
from utils import get_log_dir

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'training_data',
    '../data/tfrecords/jet_training_8101_pT-ALL_eta-ALL_Pythia.tfrecords',
    'the training data set'
)
tf.app.flags.DEFINE_string(
    'validation_data',
    '../data/tfrecords/jet_validation_2701_pT-ALL_eta-ALL_Pythia.tfrecords',
    'the validation data set'
)
tf.app.flags.DEFINE_integer('batch_size', 500, 'batch size')
tf.app.flags.DEFINE_integer('num_epochs', 30, 'the number of epochs')
tf.app.flags.DEFINE_float('initial_lr', 0.001, 'ininital learning rate')
tf.app.flags.DEFINE_float('dropout_prob', 0.5, 'the probability of dropout')

log_dir = get_log_dir(dname='test', creation=True)

train(
    tfrecords_path=FLAGS.training_data,
    tfevents_dir=log_dir.tfevents.training.path,
    ckpt_dir=log_dir.ckpt.path,
    benchmark_path=log_dir.path,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs,
    initial_lr=FLAGS.initial_lr,
    dropout_prob=FLAGS.dropout_prob,
)

evaluate(
    training_data=FLAGS.training_data,
    validation_data=FLAGS.validation_data,
    log_dir=log_dir
)

draw_all_qg_histograms(qg_histogram_dir=log_dir.qg_histogram)