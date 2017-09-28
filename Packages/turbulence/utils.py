from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_date():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M')[2:]


def make_dir(dirpath, del_recur=True):
    if del_recur:
        if tf.gfile.Exists(dirpath):
            print('Delete the existing directory.')
            tf.gfile.DeleteRecursively(dirpath)
    print('Create a directory with the following path.')
    print(dirpath)
    tf.gfile.MakeDirs(dirpath)



class Directory(object):
    def __init__(self, path, creation=True):
        self.path = path
        if creation:
            tf.gfile.MakeDirs(path)

    def make_subdir(self, dname, creation=True):
        subdpath = os.path.join(self.path, dname)
        setattr(self, dname, Directory(subdpath))
        if creation:
            tf.gfile.MakeDirs(subdpath)


def get_log_dir(dpath, creation=True):
    # mkdir
    log = Directory(dpath)
    log.make_subdir('tfevents', creation)
    log.make_subdir('ckpt', creation)
    log.make_subdir('validation', creation)
    log.make_subdir('roc', creation)
    log.make_subdir('histogram', creation)
    return log


class CkptParser(object):
    def __init__(self, path):
        with open(os.path.join(path, 'checkpoint')) as f:
            lines = f.readlines()[1:]
        self.ckpt_list = []
        for l in lines:
            name = l.split('"')[1]
            step = int(name.split('-')[1])
            self.ckpt_list.append({'path': os.path.join(path, name), 'step': step})
        
        self.step_list = self._make_step_list()
        self.path_list = self._make_path_list()

    def _make_step_list(self):
        step_list = []
        for d in self.ckpt_list:
            step_list.append(d['step'])
            step_list.sort()
        return step_list

    def _make_path_list(self):
        path_list = []
        for step in self.step_list:
            path = self.get_path(step)
            path_list.append(path)
        return path_list

    def get_path(self, step):
        output = filter(lambda d: d['step'] == step, self.ckpt_list)
        if len(output) != 1:
            raise ValueError('')
        return output[0]['path']
   

def get_available_device(device_type=None):
    ''' ref. https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow '''
    local_device_protos = device_lib.list_local_devices()
    if device_type:
        device_list = [x.name for x in local_device_protos if x.device_type == device_type]
    else:
        device_list = [x.name for x in local_device_protos]
    return device_list 


def _calc_num_examples_in_a_single_file(filename):
    return sum(1 for _ in tf.python_io.tf_record_iterator(filename))


def count_num_examples(filenames):
    if isinstance(filenames, list):
        count = 0
        for filename in filenames:
            count += _calc_num_examples_in_a_single_file(filename)
        return count
    elif isinstance(filenames, str):
        return _calc_num_examples_in_a_single_file(filenames)
    else:
        NotImplementedError("")


