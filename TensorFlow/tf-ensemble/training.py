from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
from datetime import datetime

import tensorflow as tf

from jet_model import EnsembleModel
from pipeline import get_filenames_dict, get_dataset

sys.path.append('../turbulence')
from losses import calc_xentropy
import optims
from metrics import calc_accuracy, Meter 
from utils import get_log_dir

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'data_dir',
    '../data/tfrecords_format/dataset_wsc_ensemble_1191187/',
    'the directory path of dataset'
)
tf.app.flags.DEFINE_string(
    'log_dir',
    './logs/log-%s' % datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
    'the directory path of dataset'
)

# hyperparameter
tf.app.flags.DEFINE_integer('tr_batch_size', 500, '')
tf.app.flags.DEFINE_integer('val_batch_size', 500, '')
tf.app.flags.DEFINE_integer('num_epochs', 60, '')

# freq
tf.app.flags.DEFINE_integer('validation_freq', 20, '')
tf.app.flags.DEFINE_integer('print_freq', 100, '')
tf.app.flags.DEFINE_integer('save_freq', 1000, '')



def train(training_files,
          validation_files,
          validation_dir,
          ckpt_dir,):
    start = time.time()
    with tf.Graph().as_default():
        # input pipeline
        training_dataset = get_dataset(filenames=training_files, batch_size=FLAGS.tr_batch_size)
        validation_dataset = get_dataset(filenames=validation_files, batch_size=FLAGS.val_batch_size)

        handle = tf.placeholder(tf.string, shape=[])

        iterator = tf.contrib.data.Iterator.from_string_handle(
            string_handle=handle,
            output_types=training_dataset.output_types,
            output_shapes=training_dataset.output_shapes
        )

        batch = iterator.get_next()

        # training_iterator = training_dataset.make_one_shot_iterator()
        #validation_iterator = validation_dataset.make_one_shot_iterator()
        training_iterator = training_dataset.make_initializable_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()


        is_training = tf.placeholder(tf.bool, name='is_training')

        units_list = [10, 20, 20, 10]
        filters_list = [8, 16]
        model = EnsembleModel(
            is_training=is_training,
            units_list=units_list,
            filters_list=filters_list
        )

        logits = model.forward_pass(
            images=batch['image'],
            variables=batch['variables']
        )
        prediction = tf.nn.softmax(logits)

        #
        #loss = tf.losses.softmax_cross_entropy(
        #    onehot_labels=batch['label'],
        #    logits=logits
        # )
        loss = tf.losses.absolute_difference(
            labels=batch['label'],
            predictions=prediction
        )
        # loss = tf.reduce_mean(loss)

        loss += 2e-4*tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
        )



        accuracy = calc_accuracy(logits, labels=batch['label'])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optims.training(loss=loss, lr=0.0015)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
                                   
        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True
            )
        )

        sess.run(init_op)

        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
                                   
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(max_to_keep=100000)

        meter = Meter(dpath=validation_dir)

        step = 0
        for epoch in range(FLAGS.num_epochs):
            sess.run(training_iterator.initializer)
            sess.run(validation_iterator.initializer)
            while True:
                try:
                    # training + validation
                    if(step%FLAGS.validation_freq==0):
                        validation_fetches = [loss, accuracy]
                        validation_feed = {handle: validation_handle, is_training: False}
                        val_loss_value, val_acc_value = sess.run(
                            fetches=validation_fetches,
                            feed_dict=validation_feed
                        )

                        training_fetches = [train_op, loss, accuracy]
                        training_feed = {handle: training_handle, is_training: True}
                        _, tr_loss_value, tr_acc_value = sess.run(
                            fetches=training_fetches,
                            feed_dict=training_feed
                        )

                        if(step%FLAGS.print_freq==0):
                            print('Epoch [%d/%d] / Step: %d' % (epoch+1, FLAGS.num_epochs, step+1))
                            print('  Training:')
                            print('    Loss %.3f | Acc. %.3f' % (tr_loss_value, tr_acc_value))
                            print('  Validation:')
                            print('    Loss %.3f | Acc. %.3f\n' % (val_loss_value, val_acc_value))

                        meter.append(
                            step=step,
                            tr_loss=tr_loss_value,
                            val_loss=val_loss_value,
                            tr_acc=tr_acc_value,
                            val_acc=val_acc_value,
                        )
                    # only training
                    else:
                        training_fetches = [train_op]
                        training_feed = {handle: training_handle, is_training: True}
                        _ = sess.run(
                            fetches=training_fetches,
                            feed_dict=training_feed
                        )

                    step += 1

                    # save
                    if(step%FLAGS.save_freq==0):
                        ckpt_path = os.path.join(ckpt_dir, 'step')
                        saver.save(sess, ckpt_path, global_step=step)

                except tf.errors.OutOfRangeError:
                    break
        meter.finish()
        coord.request_stop()
        coord.join(threads)
        sess.close()

        duration = time.time() - start
        H = duration // 3600
        M = (duration%3600) // 60
        print("Duration: %.f h %.f min" % (H, M))


def main():
    filenames_dict = get_filenames_dict(FLAGS.data_dir)

    log_dir = get_log_dir(dpath=FLAGS.log_dir)

    train(
        training_files=filenames_dict['training'],
        validation_files=filenames_dict['validation'],
        validation_dir=log_dir.validation.path,
        ckpt_dir=log_dir.ckpt.path,
    )

    print(":)")

if __name__ == "__main__":
    main()
