from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np


class DataLoader(object):
    def __init__(self, path, batch_size, key="jet"):
        self._batch_size = batch_size
        
        self.root_file = ROOT.TFile.Open(path)
        self.tree = self.root_file.Get(key)
        self._num_total_example = int(self.tree.GetEntries())
        
    def __len__(self):
        return self._num_total_example
    
    def __iter__(self):
        return self
    
    def next(self):
        while True:
            for start in xrange(0, self._num_total_example, self._batch_size):
                x = []
                y = []
                for idx in xrange(start, start+self._batch_size):
                    self.tree.GetEntry(idx)
                    x.append(
                        np.array(self.tree.image, dtype=np.float32).reshape(3, 33, 33))
                    y.append(np.int64(self.tree.label[1]))
                x = np.array(x)
                y = np.array(y)
                return (x, y)
        self.root_file.Close()
        
    def __next__(self):
        return self.next()

def generate_from_root_file(path, key="jet", batch_size=100):
    root_file = ROOT.TFile(path, "READ")
    tree = root_file.Get(key)
    num_total_example = tree.GetEntries()
    while True:
        for start in xrange(0, num_total_example, batch_size):
            x = []
            y = []
            for idx in xrange(start, start+batch_size):
                tree.GetEntry(idx)
                x.append(
                    np.array(tree.image, dtype=np.float32).reshape(3, 33, 33))
                y.append(np.int64(tree.label[1]))
            x = np.array(x)
            y = np.array(y)
            yield (x, y)
    root_file.Close()


def _non_cyclic_data_loader(path, batch_size, key):
    root_file = ROOT.TFile(path, "READ")
    tree = root_file.Get(key)
    num_total_example = tree.GetEntries()
    for start in xrange(0, num_total_example, batch_size):
        x = []
        y = []
        for idx in xrange(start, start+batch_size):
            tree.GetEntry(idx)
            x.append(
                np.array(tree.image, dtype=np.float32).reshape(3, 33, 33))
            y.append(np.int64(tree.label[1]))
        x = np.array(x)
        y = np.array(y)
        yield (x, y)
    root_file.Close()

def _cyclic_data_loader(path, batch_size, key):
    root_file = ROOT.TFile(path, "READ")
    tree = root_file.Get(key)
    num_total_example = tree.GetEntries()
    while True:
        for start in xrange(0, num_total_example, batch_size):
            x = []
            y = []
            for idx in xrange(start, start+batch_size):
                tree.GetEntry(idx)
                x.append(
                    np.array(tree.image, dtype=np.float32).reshape(3, 33, 33))
                y.append(np.int64(tree.label[1]))
            x = np.array(x)
            y = np.array(y)
            yield (x, y)
    root_file.Close()



def make_data_loader(path, batch_size, cyclic, key="jet"):
    if cyclic:
        return _cyclic_data_loader(path=path, batch_size=batch_size, key=key)
    else:
        return _non_cyclic_data_loader(path=path, batch_size=batch_size, key=key)
