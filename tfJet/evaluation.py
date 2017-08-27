from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from zoo import maxout_network as inference
import model
from pipeline import inputs
from eval_utils import ROC
from qg_histogram import QGHistogram
from vis_cnn import visualize_kernels
from utils import ckpt_parser
from utils import get_log_dir
from qg_histogram import draw_all_qg_histograms


def eval_once(ckpt_path,
              tfrecords_path,
              tfevents_dir,
              roc_dir,
              qg_histogram_dir,
              training_step,
              is_training_data=False):
    with tf.Graph().as_default():
        with tf.device('/cpu:0'): 
            with tf.name_scope('input'):
                dataset = inputs(data_path_list=tfrecords_path, batch_size=500, num_epochs=1)
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
        logits = inference(images=dataset['image'], keep_prob=keep_prob)
        prediction = tf.nn.softmax(logits)
        loss = model.loss(logits, dataset['label'])
        accuracy = model.evaluation(logits, dataset['label'])
        if not is_training_data:
            visualize_kernels()
        # Session
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(tfevents_dir)
            saver = tf.train.Saver()
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            saver.restore(sess, ckpt_path)
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            roc = ROC(
                dpath=roc_dir,
                step=training_step,
                title='Quark/Gluon Discrimination'
            )
            qg_histogram = QGHistogram(
                dpath=qg_histogram_dir,
                step=training_step,
                is_training_data=is_training_data
            )
            step = 0
            try:
                while not coord.should_stop():
                    labels_np, preds_np, nMatchedJets_np, loss_value, acc_value = sess.run(
                        [dataset['label'], prediction, dataset['nMatchedJets'], loss, accuracy],
                        feed_dict={keep_prob: 1.0}
                    )
                    # roc curve
                    roc.append_data(labels=labels_np[:, 0], preds=preds_np[:, 0])
                    # quark gluon histogram
                    qg_histogram.fill(labels=labels_np, preds=preds_np, nMatchedJets=nMatchedJets_np)
                    if step % 20 == 0:
                        summary = sess.run(merged, feed_dict={keep_prob: 1.0})
                        writer.add_summary(summary, step)
                        print('(step: %d) loss = %.3f, acc = %.3f' % (step, loss_value, acc_value))
                    step += 1
            except tf.errors.OutOfRangeError:
                print('%d steps' % step)
            finally:
                roc.finish()
                qg_histogram.finish()
                coord.request_stop()
                coord.join(threads)


def evaluate(log_dir,
             training_data,
             validation_data,):

    ckpt_list = ckpt_parser(log_dir.ckpt.path)
    for ckpt in ckpt_list:
        subdname = 'step_%s' % str(ckpt['step']).zfill(6)
        log_dir.tfevents.make_subdir(subdname)
        subd = getattr(log_dir.tfevents, subdname)
        subd.make_subdir('training')
        subd.make_subdir('validation')

        # on training set
        eval_once(
            training_step=ckpt['step'],
            ckpt_path=ckpt['path'],
            tfrecords_path=training_data,
            tfevents_dir=subd.training.path,
            is_training_data=True,
            roc_dir=log_dir.roc.training.path,
            qg_histogram_dir=log_dir.qg_histogram.training.path,
        )
        # on validation set
        eval_once(
            training_step=ckpt['step'],
            ckpt_path=ckpt['path'],
            tfrecords_path=validation_data,
            tfevents_dir=subd.validation.path,
            is_training_data=False,
            roc_dir=log_dir.roc.validation.path,
            qg_histogram_dir=log_dir.qg_histogram.validation.path,
        )


def main():
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
    tf.app.flags.DEFINE_string(
        'dname',
        'maxout_test',
        'the name of directory having TF checkpoint files..'
    )

    log_dir = get_log_dir(dname='maxout_test', creation=False)

    evaluate(
        training_data=FLAGS.training_data,
        validation_data=FLAGS.validation_data,
        log_dir=log_dir
    )

    draw_all_qg_histograms(qg_histogram_dir=log_dir.qg_histogram)


if __name__ == '__main__':
    main()