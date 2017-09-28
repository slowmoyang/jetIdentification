from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
from datetime import datetime
import argparse

import tensorflow as tf

from jet_model import DenseModel
from pipeline import get_filenames_dict, get_dataset
from evaluation import evaluate


sys.path.append("../../Packages")
from turbulence.losses import calc_xentropy
from turbulence.losses import calc_weak_loss
import turbulence.optims as optims
from turbulence.metrics import calc_accuracy, Meter 
from turbulence.utils import get_log_dir
from turbulence.utils import count_num_examples
from turbulence.utils import CkptParser


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train(training_files,
          validation_files,
          log_dir,
          args):
    start = time.time()
    with tf.Graph().as_default():
        global_step = tf.get_variable(
            name='global_step',
            shape=[],
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        #####################################
        # input pipeline
        #####################################
        training_dataset = get_dataset(
            filenames=training_files, batch_size=args.batch_size)
        validation_dataset = get_dataset(
            filenames=validation_files, batch_size=args.val_batch_size)

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


        ###############################################
        #                  MODEL
        ###############################################
        is_training = tf.placeholder(tf.bool, name='training_mode')

        units_list = [10, 25, 50, 100, 200, 100, 50, 25, 10, 2]
        seq = ["Dense", "ReLU", "BN"]
        model = DenseModel(
            units_list=units_list,
            seq=seq,
            is_training=is_training
        )
        model.write_model_params(dpath=log_dir.path)

        logits = model.forward_pass(batch['variables'])

        predictions = tf.nn.softmax(logits)

        loss = calc_weak_loss(
            weak_labels=batch['label_weak'], predictions=predictions)

        ############################################################
        #                     METRICS
        ##########################################################
        xentropy = calc_xentropy(
            logits=logits, labels=batch['label'])


        accuracy = calc_accuracy(logits, labels=batch['label'])
        tf.summary.scalar("accuracy", accuracy)

        meter = Meter(
            data_name_list=["step", "tr_acc", "val_acc",
                            "tr_weak_loss",
                            "tr_xentropy", "val_xentropy"],
            dpath=log_dir.validation.path
        )
        meter.prepare(
            data_pair_list=[("step", "tr_acc"), ("step", "val_acc")],
            title="Accuracy"
        )
        meter.prepare(
            data_pair_list=[("step", "tr_weak_loss")],
            title="WeakLoss(criterion)"
        )
        meter.prepare(
            data_pair_list=[("step", "tr_xentropy"), ("step", "val_xentropy")],
            title="Cross-Entropy(metric)"
        )


        #######################################################
        #                 OPTIMIZATION ALGORITHM
        ######################################################
        num_examples_per_epoch_for_train = count_num_examples(
            filenames=training_files
        )

        lr = optims.get_decayed_lr(
            global_step=global_step,
            num_examples_per_epoch_for_train=num_examples_per_epoch_for_train,
            batch_size=args.batch_size,
            num_epochs_per_decay=args.num_epochs_per_decay,
            initial_learning_rate=args.initial_learning_rate,
            learning_rate_decay_factor=args.learning_rate_decay_factor
        )



        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optims.training(
                loss=loss, lr=lr,
                optimizer=tf.train.AdagradOptimizer
            )

        ###################################################
        #
        ###################################################
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
                                   
        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True
            )
        )

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir.tfevents.path, sess.graph)

        sess.run(init_op)

        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
                                   
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(max_to_keep=100000)


        step = 0
        for epoch in range(args.num_epochs):
            sess.run(training_iterator.initializer)
            sess.run(validation_iterator.initializer)
            while True:
                try:
                    # training + validation
                    if(step%args.validation_freq==0):
                        validation_fetches = [xentropy, accuracy]
                        validation_feed = {handle: validation_handle, is_training: False}
                        val_xentropy_value, val_acc_value = sess.run(
                            fetches=validation_fetches,
                            feed_dict=validation_feed
                        )

                        training_fetches = [merged, train_op, loss, xentropy, accuracy]
                        training_feed = {handle: training_handle, is_training: True}
                        # Record execution stats
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, _, tr_weak_loss_value, tr_xentropy_value, tr_acc_value = sess.run(
                            fetches=training_fetches,
                            feed_dict=training_feed,
                            options=run_options,
                            run_metadata=run_metadata
                        )
                        writer.add_summary(summary, step)

                        if(step%args.print_freq==0):
                            print('Epoch [%d/%d] / Step: %d' % (epoch+1, args.num_epochs, step+1))
                            print('  Training:')
                            print('    Weak Loss %.3f' % (tr_weak_loss_value))
                            print('    Cross-Entropy %.3f | Acc. %.3f' % (tr_xentropy_value[0], tr_acc_value))
                            print('  Validation:')
                            print('    Cross-Entropy %.3f | Acc. %.3f\n' % (val_xentropy_value[0], val_acc_value))

                        meter_dict = {
                            "step": step,
                            "tr_weak_loss": tr_weak_loss_value,
                            "tr_xentropy": tr_xentropy_value[0],
                            "val_xentropy": val_xentropy_value[0],
                            "tr_acc": tr_acc_value,
                            "val_acc": val_acc_value
                        }
                        meter.append(data_dict=meter_dict)
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
                    if(step%args.save_freq==0):
                        ckpt_path = os.path.join(log_dir.ckpt.path, 'step')
                        saver.save(sess, ckpt_path, global_step=step)

                except tf.errors.OutOfRangeError:
                    break
        writer.close()
        meter.finish()
        # When done, ask the threads to stop.
        coord.request_stop()
        # Wait for threads to fininsh.
        coord.join(threads)
        sess.close()

        duration = time.time() - start
        H = duration // 3600
        M = (duration%3600) // 60
        print("Duration: %.f h %.f min" % (H, M))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
	'--data_dir', type=str,
        default='../../data/tfrecords_format/dataset_variables_13310',
	help='the directory path of dataset'
    )

    parser.add_argument(
	'--log_dir', type=str,
	default='./logs/weak-%s' % datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
	help='the directory path of dataset'
    )

    # hyperparameter
    parser.add_argument('--batch_size', type=int, default=2000, help='')
    parser.add_argument('--val_batch_size', type=int, default=500, help='')
    parser.add_argument('--num_epochs', type=int, default=1, help='')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999,
			help='The decay to use for the moving average.')
    parser.add_argument("--num_epochs_per_decay", type=int, default=350,
			help="Epochs after which learning rate decays.")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.1,
			help="Learning rate decay factor.")
    parser.add_argument("--initial_learning_rate", type=float, default=0.1,
                        help="Initial learning rate.")
    # freq
    parser.add_argument('--validation_freq', type=int, default=20, help='')
    parser.add_argument('--print_freq', type=int, default=100, help='')
    parser.add_argument('--save_freq', type=int, default=1000, help='')

    args = parser.parse_args()

    filenames_dict = get_filenames_dict(args.data_dir)

    log_dir = get_log_dir(dpath=args.log_dir)

    train(
        training_files=filenames_dict['training'],
        validation_files=filenames_dict['validation'],
        log_dir=log_dir,
        args=args
    )

    # checkpoint
    ckpt_parser = CkptParser(log_dir.ckpt.path)
    for step, ckpt_path in zip(ckpt_parser.step_list, ckpt_parser.path_list):
        evaluate(
            ckpt_path=ckpt_path,
            training_step=step,
            test_filenames=filenames_dict['test'],
            log_dir=log_dir
        )

    print(":)")


if __name__ == "__main__":
    main()
