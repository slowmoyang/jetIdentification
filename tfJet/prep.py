from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import ROOT
import numpy as np
import tensorflow as tf

from prep_utils import fill
from prep_utils import convert_to_feature
from utils import Directory


create_dataset = lambda: {
    'image': [],
    'label': [],
    'pT': [],
    'eta': [],
    'nMatchedJets': [],
    'nJets': [],
    'nGenJets': [],
    'partonId': [],
}


def root_to_np(input_path,
               eta_threshold=2.4,
               C=3, H=33, W=33,
               deta_max=0.4, dphi_max=0.4,
               tree_name="jetAnalyser/jetAnalyser"):
    '''
    NCHW format
    N : the number of jet
    C : channel (0: cpt, 1: npt, 2: cmu)
    H : height (~
    W : width (~the number of column)

    To make data without eta binning, set eta_threshold to np.inf

    '''
    tfile = ROOT.TFile(input_path, "READ")
    jet = tfile.Get(tree_name)
    entries = jet.GetEntriesFast()

    above = create_dataset()
    below = create_dataset()

    start = time.time()
    for j in xrange(entries):
        if j % 1000 == 0:
            print('( %s ) %dth jet' % (time.asctime(), j))

        jet.GetEntry(j)

        # eta below threshold value (defalult 2.4)
        if abs(jet.eta) < eta_threshold:
            # gluon: background
            if jet.partonId == 21:
                below['label'].append([[0, 1]])
            elif jet.partonId in [1, 2, 3]:
                below['label'].append([[1, 0]])
            else:
                continue

            img = np.zeros(shape=(C, H, W), dtype=np.float32)
            for d in xrange(len(jet.dau_pt)):
                if (-deta_max < jet.dau_deta[d] < deta_max) and (-dphi_max < jet.dau_dphi[d] < dphi_max):
                    # neutral particle
                    if jet.dau_charge[d]:
                        # pT
                        fill(img[1], jet.dau_deta[d], jet.dau_dphi[d], jet.dau_pt[d], H)
                    # charged particle
                    else:
                        # pT
                        fill(img[0], jet.dau_deta[d], jet.dau_dphi[d], jet.dau_pt[d], H)
                        # multiplicity
                        fill(img[2], jet.dau_deta[d], jet.dau_dphi[d], 1, H)

            # collect data
            keys = ['image', 'pT', 'eta', 'nMatchedJets', 'nJets', 'nGenJets', 'partonId']
            items = [img, jet.pt, jet.eta, jet.nMatchedJets, jet.nJets, jet.nGenJets, jet.partonId]
            for k, i in zip(keys, items):
                below[k].append(i)

            below['eta_range'] = 'below_%.1f' % eta_threshold
            below['pT_range'] = 'ALL'

        # eta above threshold value
        else:
            if jet.partonId == 21:
                above['label'].append([[0, 1]])
            elif jet.partonId in [1, 2, 3]:
                above['label'].append([[1, 0]])
            else:
                continue

            img = np.zeros(shape=(1, H, W), dtype=np.float32)
            for d in xrange(len(jet.dau_pt)):
                if (-deta_max < jet.dau_deta[d] < deta_max) and (-dphi_max < jet.dau_dphi[d] < dphi_max):
                    fill(img[0], jet.dau_deta[d], jet.dau_dphi[d], jet.dau_pt[d], H)

            # collect data
            keys = ['image', 'pT', 'eta', 'nMatchedJets', 'nJets', 'nGenJets', 'partonId']
            items = [img, jet.pt, jet.eta, jet.nMatchedJets, jet.nJets, jet.nGenJets, jet.partonId]
            for k, i in zip(keys, items):
                above[k].append(i)

            above['eta_range'] = 'above_%.1f' % eta_threshold
            above['pT_range'] = 'ALL'

    duration = time.time() - start
    duration /= 60

    for ds in [below, above]:
        for k in ds.keys():
            if isinstance(ds[k], list):
                ds[k] = np.array(ds[k])

    print('duration: %.1f min' % duration)
    print('entries in root file: %d' % entries)
    print('#examples with eta below %.1f: %d' % (eta_threshold, below['image'].shape[0]))
    print('#examples with eta above %.1f: %d' % (eta_threshold, above['image'].shape[0]))

    return below, above


def pt_binning(dataset, minimum=0.0, maximum=np.inf):
    condition = np.logical_and(
        dataset['pT'] < maximum,
        dataset['pT'] > minimum
    )

    indices = np.where(condition)

    new_dataset = {}

    for k in dataset.keys():
        if isinstance(dataset[k], list):
            new_dataset[k] = dataset[k][indices]

    new_dataset['eta_range'] = dataset['eta_range']
    new_dataset['pT_range'] = '%d-%d' % (minimum, maximum)

    return new_dataset


def np_to_tfrecords(dataset, output_path):
    # Open the TFRecords file
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in xrange(dataset['image'].shape[0]):
        # only length-1 array can be converted to Python scalars.
        image_raw = dataset['image'][i].tostring()
        label_raw = dataset['label'][i].tostring()
        feature = {
            # Convert data into the proper data type of the feature using tf.traid.<DATA_TYPE>List
            'image': convert_to_feature(value=image_raw, dtype='bytes'),
            'label': convert_to_feature(value=label_raw, dtype='bytes'),
            'pT': convert_to_feature(value=dataset['pT'][i], dtype='float'),
            'eta': convert_to_feature(value=dataset['eta'][i], dtype='float'),
            'nMatchedJets': convert_to_feature(value=dataset['nMatchedJets'][i], dtype='int'),
            'partonId': convert_to_feature(value=dataset['partonId'][i], dtype='int'),
            'nJets': convert_to_feature(value=dataset['nJets'][i], dtype='int'),
            'nGenJets': convert_to_feature(value=dataset['nGenJets'][i], dtype='int'),
        }
        # Create a feature using tf.train.Feature and pass the converted data to it.
        # Create an Example protocol buffer using tf.train.Example and pass the converted data to it.
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize the Example to string using example.SerializeToString()
        # Write the serialized example to TFRecords file using writer.write
        writer.write(example.SerializeToString())
    # Close the file
    writer.close()

 
def split_n_save(dataset, generator, data_path='../data'):
    total = dataset['image'].shape[0]

    dname = 'jet_pT-%s_eta-%s_%s' % (dataset['pT_range'], dataset['eta_range'], generator)
    output_dpath = os.path.join(data_path, dname)
    output_dir = Directory(output_dpath)
    output_dir.make_subdir('npz')
    output_dir.make_subdir('tfrecords')

    # start index for each data set
    # i.e. training_idx = o
    val_idx = int(total*0.6)
    test_idx = int(total*0.8)

    idx_list = [(0, val_idx), (val_idx, test_idx), (test_idx, None)]
    tag_list = ['training', 'validation', 'test']

    for (start_idx, end_idx), tag in zip(idx_list, tag_list):
        # data set to save
        ds = {}
        for k in dataset.keys():
            if isinstance(dataset[k], np.ndarray):
                ds[k] = dataset[k][start_idx: end_idx]

        num_example = ds['image'].shape[0]

        fname = 'jet_%s_%d' % (tag, num_example)

        npz_path = os.path.join(output_dir.npz.path, fname)
        np.savez(
            npz_path,
            image=ds['image'],
            label=ds['label'],
            pT=ds['pT'],
            eta=ds['eta'],
            nMatchedJets=ds['nMatchedJets'],
            partonId=ds['partonId'],
            nJets=ds['nJets'],
            nGenJets=ds['nGenJets'],
        )

        # Convert numpy array (ndarray object) to .tfrecords
        tfrecords_path = os.path.join(output_dir.tfrecords.path, fname+'.tfrecords')
        np_to_tfrecords(dataset=ds, output_path=tfrecords_path)


def main():
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string(
        'input_path',
        '../data/root/jet_pythia_1.root',
        ''
    )
    tf.app.flags.DEFINE_string(
        'data_dir',
        '../data/',
        ''
    )
    tf.app.flags.DEFINE_integer('eta_threshold', 2.4, 'if value is -1, no eta binning')
    
    dataset_list = []
    if FLAGS.eta_threshold == -1:
        ds = root_to_np(input_path=FLAGS.input_path, eta_threshold=np.inf)
        dataset_list.append(ds)
    else:
        below, above = root_to_np(input_path=FLAGS.input_path, eta_threshold=FLAGS.eta_threshold)
        dataset_list.append(below)
        dataset_list.append(above)

    for ds in dataset_list:
        split_n_save(ds, generator='pythia')


if __name__ == '__main__':
    main()
