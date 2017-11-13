from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import json
import ROOT
import numpy as np
import tensorflow as tf
import argparse


sys.path.append("../TensorFlow")
from turbulence.prep_utils import convert_to_feature
from turbulence.utils import Directory


def convert_to_tfrecords(input_path, output_path, with_weak=True, with_image=True):
    def _feature(x, dtype, shape=None):
        if shape is None:
            if hasattr(x, "__len__"):
                x = np.array(x, dtype=dtype)
                x = x.tostring()
            # if x is scalar
            else:
                x = dtype(x)
        else:
            x = np.array(x, dtype=dtype)
            x = x.reshape(shape)
            x = x.tostring()

        return convert_to_feature(x)

    root_file = ROOT.TFile(input_path, "READ")
    key = root_file.GetListOfKeys().At(0).GetName()
    tree = root_file.Get(key)
    
    writer = tf.python_io.TFRecordWriter(output_path)
    
    for i in xrange(tree.GetEntries()):
        tree.GetEntry(i)

        feature = dict()

        if with_image:
            feature["image"] = _feature(tree.image, np.float32, shape=(3, 33, 33))

        feature["variables"] = _feature(tree.variables, np.float32)
        feature["label"] = _feature(tree.label, np.int64)

        if with_weak:
            feature["label_weak"] = _feature(tree.label_weak, np.float32)

        feature["nMatchedJets"] = _feature(tree.nMatchedJets, np.int64)
        
        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )


        # write: Write a string record to the file.
        #
        # write(record)
        #   Args:
        #     * record: str
        writer.write(example.SerializeToString())
        
    writer.close()
    root_file.Close()

    
def convert_to_tfrecords_fast_sim(input_path, output_path, with_image=True):
    def _feature(x, dtype, shape=None):
        if shape is None:
            if hasattr(x, "__len__"):
                x = np.array(x, dtype=dtype)
                x = x.tostring()
            # if x is scalar
            else:
                x = dtype(x)
        else:
            x = np.array(x, dtype=dtype)
            x = x.reshape(shape)
            x = x.tostring()

        return convert_to_feature(x)

    root_file = ROOT.TFile(input_path, "READ")
    key = root_file.GetListOfKeys().At(0).GetName()
    tree = root_file.Get(key)
    
    writer = tf.python_io.TFRecordWriter(output_path)
    
    for i in xrange(tree.GetEntries()):
        tree.GetEntry(i)

        feature = dict()

        if with_image:
            feature["image"] = _feature(tree.image, np.float32, shape=(3, 33, 33))

        feature["variables"] = _feature(tree.variables, np.float32)
        feature["label"] = _feature(tree.label, np.int64)
        feature["pT"] = _feature(tree.pt, np.float32)
        feature["eta"] = _feature(tree.eta, np.float32)
        
        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )


        # write: Write a string record to the file.
        #
        # write(record)
        #   Args:
        #     * record: str
        writer.write(example.SerializeToString())
        
    writer.close()
    root_file.Close()
