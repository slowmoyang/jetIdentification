from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json

import tensorflow as tf

sys.path.append("../../Packages")
from turbulence.model_base import ModelBase


class MLWModel(ModelBase):
    def __init__(self, is_training, dropout_rate, filters_list=None):
        super(MLWModel, self).__init__()
        self._filters_list = filters_list

        # placeholders
        self._is_training = is_training
        self._dropout_rate = dropout_rate
        
    def forward_pass(self, x):
        for filters in self._filters_list[:-1]:
            x = self._make_block(x, filters)
            x = self._max_pool(x)

        x = self._make_block(x, filters=self._filters_list[-1])
        x = self._gap(x)

        x = self._maxout(x, units=10, extractors=5, name='maxout1')
        x = self._dropout(x, rate=self._dropout_rate, is_training=self._is_training)

        x = self._maxout(x, units=2, extractors=5, name='maxout2')
        return x

    def _make_block(self, x, filters):
        x = self._bn(x, is_training=self._is_training, axis=1)
        x = tf.nn.relu(x)
        x = self._conv(x, filters)
        x = self._bn(x, is_training=self._is_training, axis=1)
        x = tf.nn.relu(x)
        x = self._conv(x, filters)
        return x

    def set_filters_list(self, filters_list):
        self._filters_list = filters_list

    def write_model_params(self, dpath):
        model_params = {
            'filters_list': self._filters_list,
        }
        data = json.dumps(model_params)
        path = os.path.join(dpath, 'model_params.json')
        with open(path, 'w') as f:
            f.write(data)

    def load_model_params(self, path):
        path = os.path.join(path, 'model_params.json')
        data = open(path).read()
        data = json.loads(data)
        self.set_filters_list(data['filters_list'])



def main():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 3, 33, 33])
    print(logits)

if __name__ == "__main__":
    main()
