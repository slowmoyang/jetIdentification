from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import json

class Directory(object):
    def __init__(self, path, creation=True):
        self.path = path
        self._creation = creation
        if self._creation:
            os.mkdir(self.path)

    def make_subdir(self, name):
        path = os.path.join(self.path, name)
        setattr(self, name, Directory(path, creation=self._creation))


def get_log_dir(path, creation=True):
    # mkdir
    log = Directory(path, creation=creation)
    log.make_subdir('validation')
    log.make_subdir('saved_models')
    log.make_subdir('roc')
    log.make_subdir('output_histogram')
    return log


def get_saved_model_paths(dpath):
    def foo(f):
        step = int(f.split("_")[1].split(".")[0])
        path = os.path.join(dpath, f)
        return (path, step)

    saved_models = os.listdir(dpath)
    saved_models.sort()
    saved_models = map(foo, saved_models)
    return saved_models
    
