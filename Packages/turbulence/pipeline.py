from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


'''
ref. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

ref. https://github.com/visipedia/tfrecords/blob/master/stat_tfrecords.py
    - tf.VarLenFeature(
    - 
'''

def _parse_function(example_proto):
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
        'pT': tf.VarLenFeature(dtype=tf.float32),
        'eta': tf.VarLenFeature(dtype=tf.float32),
        'nMatchedJets': tf.FixedLenFeature([], tf.int64),
        'partonId': tf.FixedLenFeature([], tf.int64),
    }
    
    parsed_features = tf.parse_single_example(example_proto, features)
    
    # images 
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(bytes=parsed_features['image'], out_type=tf.float32) # shape=(?,)
    # The image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(tensor=image, shape=(3, 33, 33))

    # images
    label = tf.decode_raw(bytes=parsed_features['label'], out_type=tf.int64)
    label.set_shape(2)

    pT = tf.sparse_tensor_to_dense(parsed_features['pT'])
    pT.set_shape(1)

    eta = tf.sparse_tensor_to_dense(parsed_features['eta'])
    eta.set_shape(1)

    dataset = {
        'image': image,
        'label': label,
        'pT': pT,
        'eta': eta,
        'nMatchedJets': parsed_features['nMatchedJets'],
        'partonId': parsed_features['partonId'],
    }
    
    return dataset


def get_dataset(filenames, batch_size=500):
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    return dataset


def get_filenames_dict(dpath):
    output = {
        'training': [],
        'validation': [],
        'test': []
    }                
    for i in os.listdir(dpath):
        if 'training' in i:
            output['training'].append(os.path.join(dpath, i))
        elif 'validation' in i:
            output['validation'].append(os.path.join(dpath, i))
        elif 'test' in i:
            output['test'].append(os.path.join(dpath, i))
    return output

