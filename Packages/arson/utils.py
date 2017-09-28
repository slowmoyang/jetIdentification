import math
import os
import random
import torch
import time
import numpy as np


def to_numpy(x):
    if isinstance(x, torch.autograd.variable.Variable):
        x = x.data
    if hasattr(x, "cpu"):
        x = x.cpu()
    return x.numpy()

def to_var(x, cuda=True):
    if isinstance(x, torch.autograd.variable.Variable):
        raise ValueError("input is torch.autograd.variable.Variable type already :(")
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    if cuda and torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x)


def to_float(x):
    x = to_numpy(x)
    return float(x)


def load_datasets(dpath):
    files = os.listdir(dpath)
    datasets = {}
    for f in files:
        for k in ['training', 'validation', 'test']:
            datasets[k] = os.path.join(dpath, f)
            continue
    return datasets


class Directory(object):
    def __init__(self, path, creation=True):
        self.path = path
        if creation:
            os.mkdir(path)

    def make_subdir(self, dname, creation=True):
        subdpath = os.path.join(self.path, dname)
        setattr(self, dname, Directory(subdpath))
        #if creation:
        #   os.mkdir(subdpath)


def make_fake_output_n_label(batch_size=100, num_classes=2, v=True):
    output = torch.rand(batch_size, num_classes)
    c = range(num_classes)
    label = np.zeros(shape=(batch_size, num_classes), dtype=np.int64)
    for b in range(batch_size):
        i = random.choice(c) 
        label[b][i] = 1
    label = torch.Tensor(label) 
    #label = torch.Tensor([random.choice(c) for _ in range(batch_size)])
    if v:
        return to_var(output), to_var(label)
    else:
        return output, label


def calc_feature_map_size(conv, Lin, k, s=None, p=None, d=1):
    if s is None:
        s = 1 if conv else k
    if p is None:
        p = 2 if conv else 0
    Lout = int( ((Lin + 2*p - d*(k-1) -1)/s) + 1) 
    return Lout

