import os
import sys
import shutil
import platform
import time
import json
import ROOT
import numpy as np
# custom file
from utils import Directory


def make_log_dir(dname, logs_path='./logs'):
    path = os.path.join(logs_path, dname)
    if os.path.exists(path):
        shutil.rmtree(path)
    logdir = Directory(path)
    logdir.make_subdir('checkpoint')
    logdir.make_subdir('roc')
    logdir.make_subdir('output_histogram')
    return logdir

class Logger(object):
    def __init__(self, path):
        self.log = {}
        self.log['env'] = {
            'torch_version': torch.__version__,
            'os': platform.platfomr(),
            'start_time': time.asctime(),
        }
 
    def log_args(self, args):
        self.log['args'] = args.__dict__

    def finish(self):
        self.log['env']['end_time'] = time.asctime()
        self.duration = time.time() - self.start_time
        self.log['env']['duration'] = duration
         
