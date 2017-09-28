from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from zoo import maxout_network as inference
import model
from pipeline import get_filenames_dict, get_dataset
from utils import get_log_dir, CkptParser
from metric import ROC, OutputHistogram

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'data_dir',
    './data/dataset_image_wsc_1292601',
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
             roc_dir,
             histogram_dir):

    with tf.Graph().as_default():
        # Pipeline
        dataset = get_dataset(test_filenames)
        iterator = dataset.make_one_shot_iterator()
        example = iterator.get_next()
    
        # model
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)

        logits = inference(images=example['image'], keep_prob=keep_prob)
        prediction = tf.nn.softmax(logits)

        # Define loss and
        loss = model.loss(logits=logits, labels=example['label'])

        accuracy = model.evaluation(logits, labels=example['label'])

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
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
    
        roc = ROC(
            dpath=roc_dir,
            step=training_step,
            title='Quark/Gluon Discrimination'
        )
    
        histo = OutputHistogram(
            dpath=histogram_dir,
            step=training_step
        )
    
        step = 0
        try:
            while not coord.should_stop():
                fetches = [
                    example['label'], prediction, 
                    example['nMatchedJets'],
                    loss, accuracy
                ]
                feed_dict = {keep_prob: 1.0}
            
                labels_np, preds_np, nMatchedJets_np, test_loss_value, test_acc_value = sess.run(
                        fetches=fetches,
                        feed_dict=feed_dict
                )

                roc.append(labels=labels_np[:, 0],
                           preds=preds_np[:, 0])
            
                histo.fill(
                    labels=labels_np,
                    preds=preds_np,
                    nMatchedJets=nMatchedJets_np
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
            roc_dir=log_dir.roc.path,
            histogram_dir=log_dir.histogram.path
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
    main()
    #evaluate_all()
