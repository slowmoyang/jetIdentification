from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import ROOT
import numpy as np
import tensorflow as tf

from utils import Directory

def convert_to_feature(value):
    if isinstance(value, np.int64):
        return _int64_feature(value)
    elif isinstance(value, np.float32):
        return _float_feature(value)
    elif isinstance(value, str):
        # tf.compat: Functions for Python 2 vs. 3 compatibility
        # tf.compat.as_bytes: Converts either bytes or unicode to bytes, using utf-8 encoding for text.
        return _bytes_feature(tf.compat.as_bytes(value))
    else:
        raise NotImplementedError('Not implemented :(')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def root_to_tfrecords(input_path, output_path):
    root_file = ROOT.TFile(input_path, "READ")
    key = root_file.GetListOfKeys().At(0).GetName()
    tree = root_file.Get(key)
    
    writer = tf.python_io.TFRecordWriter(output_path)
    
    for i in xrange(tree.GetEntries()):
        tree.GetEntry(i)
        # example
        _image = np.array(tree.image, dtype=np.float32).reshape(3, 33, 33)
        _label = np.array(tree.label, dtype=np.int64)
        # additional info
        _nMatchedJets = np.int64(tree.nMatchedJets)
        _pt = np.float32(tree.pt)
        _eta = np.float32(tree.eta)
        _partonId = np.int64(tree.partonId)
        
        
        image_raw = _image.tostring()
        label_raw = _label.tostring()
        
        feature = {
            # Convert data into the proper data type of the feature using tf.traid.<DATA_TYPE>List
            'image': convert_to_feature(value=image_raw),
            'label': convert_to_feature(value=label_raw),
            # additional info
            'pT': convert_to_feature(value=_pt),
            'eta': convert_to_feature(value=_eta),
            'nMatchedJets': convert_to_feature(value=_nMatchedJets),
            'partonId': convert_to_feature(value=_partonId),
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        
    writer.close()
    root_file.Close()
