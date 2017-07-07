from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

def read_and_decode(filename_queue, image_shape=(3,33,33), label_shape=(2)):
    # Define a reader
    reader = tf.TFRecordReader()

    # Read the next record
    _, serialized_example = reader.read(queue=filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(
        serialized = serialized_example,
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        }
    )

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(bytes = features['image'], out_type = tf.float32) # shape=(?,)
    # The image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(tensor = image, shape = image_shape)

    # In the same way...
    label = tf.decode_raw(bytes = features['label'], out_type = tf.int64)
    label.set_shape(label_shape)

    return image, label

def inputs(data_path_list, batch_size=100, num_epochs=500):
    with tf.name_scope('input'):
        if type(data_path_list) == str:
            data_path_list = [data_path_list]
        
        # Create a queue that produces the filenames to read
        # the type of filename_queue is FIFOQueue object
        filename_queue = tf.train.string_input_producer(
            string_tensor = data_path_list,
            num_epochs = num_epochs
        )

        image, label = read_and_decode(filename_queue)

        """
        shuffle_batch: Creates batches by randomly shuffling tensors.
            Args:
              - tensors: The list or dictionary of tensors to enqueue
              - batch_size: The new
              - num_threads: The number of threads enque
              - capacity: The capacity argument controls the how long the prefetching is allowed to grow the queues.
              - min_after_dequeue: Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
            Returns:
              A list or dictionary of tensors with the types as tensors.
        """
        images, sparse_labels = tf.train.shuffle_batch(
            tensors = [image, label],
            batch_size = batch_size,
            num_threads = 2, # The number of threads enque
            capacity = 1000 + 3 * batch_size,
            min_after_dequeue = 1000
        )
    return images, sparse_labels
