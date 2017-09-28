from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json

import tensorflow as tf

sys.path.append("../../Packages")
from turbulence.model_base import ModelBase

'''
ref. https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/model_base.py 
ref. https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_model.py
ref. https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_main.py


'''

class DenseModel(ModelBase):
    def __init__(self, units_list=None, seq=None, is_training=None):
        super(DenseModel, self).__init__()
        self._units_list = units_list
        self._seq = seq
        self._is_training = is_training
        
    def forward_pass(self, x):
        for units in self._units_list:
            x = self._make_block(x, units, seq=self._seq)
        return x

    def _make_block(self, x, units, seq=['Dense', 'BN', 'ReLU']):
        def _get_layer(u, key):
            key = key.lower()
            if key == "dense":
                return self._dense(x, units)
            elif key == "bn":
                return self._bn(x, self._is_training, axis=1)
            elif key == "relu":
                return self._relu(u)
            elif key == 'selu':
                return self._selu(u)
            else:
                raise KeyError("%s is not implemented" % key)
        for k in seq:
            x = _get_layer(x, k)
        return x


    def set_units_list(self, units_list):
        self._units_list = units_list

    def set_seq(self, seq):
        self._seq = seq

    def write_model_params(self, dpath):
        model_params = {
            'units_list': self._units_list,
            'seq': self._seq
        }
        data = json.dumps(model_params)
        path = os.path.join(dpath, 'model_params.json')
        with open(path, 'w') as f:
            f.write(data)

    def load_model_params(self, path):
        path = os.path.join(path, 'model_params.json')
        data = open(path).read()
        data = json.loads(data)
        self.set_units_list(data['units_list'])
        self.set_seq(data['seq'])

def main():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 5])
    is_training = tf.placeholder(dtype=tf.bool)
    model = DenseModel(
        units_list=[100, 15, 5 ,2],
        seq=["Dense", "BN", "ReLU"],
        is_training=is_training
    )
    logits = model.forward_pass(x)
    print(logits)

if __name__ == "__main__":
    main()
