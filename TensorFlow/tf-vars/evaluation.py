from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf

from jet_model import DenseModel
from pipeline import get_filenames_dict
from pipeline import get_dataset

sys.path.append("../../Packages")
from turbulence.losses import calc_xentropy
from turbulence.metrics import calc_accuracy
from turbulence.metrics import ROC, OutputHistogram
from turbulence.metrics import OutputHistogram
from turbulence.utils import get_log_dir
from turbulence.utils import CkptParser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'data_dir',
    '../data/tfrecords_format/dataset_variables1752103',
    'the directory path of dataset'
)
tf.app.flags.DEFINE_string(
    'log_dir',
    './logs/log-2017-09-06_22-44-24',
    'the directory path of dataset'
)
tf.app.flags.DEFINE_integer('step', 1000, '')

def evaluate(ckpt_path,
             training_step,
             test_filenames,
             log_dir):

    with tf.Graph().as_default():
        # Pipeline
        dataset = get_dataset(test_filenames)
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()

        ###############################################    
        #                 MODEL
        ################################################
        is_training = tf.placeholder(tf.bool, name='training_mode')

        model = DenseModel(is_training=is_training)

        # Load model parameters
        model.load_model_params(path=log_dir.path)

        logits = model.forward_pass(batch['variables'])

        predictions = tf.nn.softmax(logits)

        loss = calc_xentropy(logits=logits, labels=batch['label'])

        ####################################################
        # Define loss and
        #####################################################
        accuracy = calc_accuracy(logits, labels=batch['label'])

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
                                   
        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
            log_device_placement=True
            )
        )

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            sess=sess, coord=coord
        )

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
    
        roc = ROC(
            dpath=log_dir.roc.path,
            step=training_step,
            title='Quark/Gluon Discrimination'
        )
    
        histo = OutputHistogram(
            dpath=log_dir.histogram.path,
            step=training_step
        )
    
        step = 0
        print('\n'*10)
        try:
            while not coord.should_stop():
                fetches = [batch['label'], predictions, 
                           batch['nMatchedJets'],
                           loss, accuracy]
                feed_dict = {is_training: False}
            
                labels_value, preds_value, nMatchedJets_value, test_loss_value, test_acc_value = sess.run(
                        fetches=fetches, feed_dict=feed_dict)

                roc.append(labels=labels_value[:, 1],
                           preds=preds_value[:, 1])
            
                histo.fill(
                    labels=labels_value,
                    preds=preds_value,
                    nMatchedJets=nMatchedJets_value
                )
            
                #print('Step: %d' % step)
                #print('  Loss %.3f | Acc. %.3f' % (test_loss_value, test_acc_value))
            
                step += 1

        except tf.errors.OutOfRangeError:
            print("Evaluation is over! :D")
        finally:
            roc.finish()
            histo.finish()
            coord.request_stop()
            coord.join(threads)
            sess.close()


def evaluate_all():
    # log directory
    log_dir = get_log_dir(dpath=FLAGS.log_dir, creation=False) 

    # data
    filenames_dict = get_filenames_dict(FLAGS.data_dir)

    # checkpoint
    ckpt_parser = CkptParser(log_dir.ckpt.path)
    for step, ckpt_path in zip(ckpt_parser.step_list, ckpt_parser.path_list):
        evaluate(
            ckpt_path=ckpt_path,
            training_step=step,
            test_filenames=filenames_dict['test'],
            log_dir=log_dir
        )
       




def main():
    # log directory
    log_dir = get_log_dir(dpath=FLAGS.log_dir, creation=False) 

    # checkpoint
    ckpt_parser = CkptParser(log_dir.ckpt.path)
    ckpt_path = ckpt_parser.get_path(step=FLAGS.step)

    # data
    filenames_dict = get_filenames_dict(FLAGS.data_dir)

    evaluate(
        ckpt_path=ckpt_path,
        training_step=FLAGS.step,
        test_filenames=filenames_dict['test'],
        roc_dir=log_dir.roc.path,
        histogram_dir=log_dir.histogram.path
    )


if __name__ == '__main__':
    # main()
    evaluate_all()
