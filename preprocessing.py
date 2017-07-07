from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import ROOT
import numpy as np
import tensorflow as tf
FLAGS = None

def foo(ndarr, deta, dphi, entry, bin=33, deta_max=0.4, dphi_max=0.4):
    if deta == deta_max:
        x_idx = bin - 1
    else:
        x = deta + deta_max
        dx = 2 * deta_max / bin
        x_idx = int(x/dx)
    
    if dphi == dphi_max:
        y_idx = bin - 1
    else:
        y = ( dphi + dphi_max )
        dy = 2 * dphi_max / bin
        y_idx = int(y/dy)
    ndarr[x_idx][y_idx] += entry

def foo2(ndarr, deta, dphi, entry, bin_num, deta_max=0.4, dphi_max=0.4):
    def _bar(dx, dx_max):
        if dx == dx_max:
            return bin_num - 1
        else:
            return int( (dx+dx_max) / (2*dx_max/bin_num) )
    row = _bar(deta, deta_max)
    col = _bar(dphi, dphi_max)
    ndarr[row][col] += entry
    

def make_data(input_path, tree_name="jetAnalyser/jetAnalyser",
              bin_num=33, deta_max=0.4, dphi_max=0.4):
    '''
    NCHW format
    N : the number of jet
    C : channel (0: cpt, 1: npt, 2: cmu)
    H : height (~
    W : width (~the number of column)

    '''
    tfile = ROOT.TFile(input_path, "READ")
    jet = tfile.Get(tree_name)
    entries = jet.GetEntriesFast()
    # NCHW format
    # N 
    images = np.zeros((entries, 3, bin_num, bin_num), dtype=np.float32) # NCHW format
    labels = np.zeros((entries, 2), dtype=np.int64)
    jetids = np.zeros((entries))
    for j in xrange(entries):
        if j / 1000 == 0:
            print('( %s ) %dth jet' % (time.asctime(), j))
        jet.GetEntry(j)
        # onehot encoding
        # gluon ~ background
        if jet.partonId == 21:
            labels[j][1] = 1
        # quark ~ signal
        else:
            labels[j][0] = 1
        # jet parton Id
        jetids[j] = jet.partonId
        for d in xrange(len(jet.dau_pt)):
            # if False not in (-0.4 < jet.deta[d], jet.dphi[d] < 0.4)
            if (-deta_max < jet.dau_deta[d] < deta_max) and (-dphi_max < jet.dau_dphi[d] < dphi_max):
                # neutral particle
                if jet.dau_charge[d]:
                    # pT
                    foo(images[j][1], jet.dau_deta[d], jet.dau_dphi[d], jet.dau_pt[d], bin_num)
                # charged particle
                else:
                    # pT
                    foo(images[j][0], jet.dau_deta[d], jet.dau_dphi[d], jet.dau_pt[d], bin_num)
                    # multiplicity
                    foo(images[j][2], jet.dau_deta[d], jet.dau_dphi[d], 1, bin_num)
    return images, labels, jetids

def load_npz(path):
    container = np.load(path)
    return [container[key] for key in container]

###########################################################################
#
#    .ndarray -----> .tfrecords
#
############################################################################

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_to_tfrecords(images, labels, output_path):
    # Open the TFRecords file
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in xrange(images.shape[0]):
        # only length-1 array can be converted to Python scalars.
        image_raw = images[i].tostring()
        label_raw = labels[i].tostring()
        feature = {
            # Convert data into the proper data type of the feature using tf.traid.<DATA_TYPE>List
            'image': _bytes_feature(tf.compat.as_bytes(image_raw)),
            'label': _bytes_feature(tf.compat.as_bytes(label_raw)),
        }
        # Create a feature using tf.train.Feature and pass the converted data to it.
        # Create an Example protocol buffer using tf.train.Example and pass the converted data to it.
        example = tf.train.Example(
            features = tf.train.Features(feature=feature))
        # Serialize the Example to string using example.SerializeToString()
        # Write the serialized example to TFRecords file using writer.write
        writer.write(example.SerializeToString())
    # Close the file
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_dir', type=str, default='../data/tfrecords/')
    parser.add_argument('--fname', type=str)
    parser.add_argument('--NHWC', type=bool, default=True)
    FLAGS, unparsed = parser.parse_known_args()
    # Load the data
    images, labels = load_npz(FLAGS.input_path)
    fname = FLAGS.fname + '_NCHW.tfreocrds'
    path = os.path.join(FLAGS.output_dir, fname)
    print("Start to write a NCHW file")
    write_to_tfrecords(images, labels, path)
    #
    if FLAGS.NHWC:
        print("NCHW ----> NHWC")
        images = images.transpose([0, 2, 3, 1])
        nhwc_fname = FLAGS.fname + '_NHWC.tfrecords' 
        nhwc_path = os.path.join(FLAGS.output_dir, nhwc_fname)
        print("Start to write a NHWC file")
        write_to_tfrecords(images, labels, nhwc_path)
