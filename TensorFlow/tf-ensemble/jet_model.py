from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

sys.path.append("../turbulence")
from model_base import ModelBase

'''
ref. https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/model_base.py 
ref. https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_model.py
ref. https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_main.py


'''

class EnsembleModel(ModelBase):
    def __init__(self, is_training, units_list, filters_list):
        super(EnsembleModel, self).__init__(is_training)
        self._is_training = is_training
        self._units_list = units_list
        self._filters_list = filters_list
        
    def forward_pass(self, images, variables):
        dense_stem = self._make_dense_stem(x=variables, units_list=self._units_list)
        conv_stem = self._make_conv_stem(x=images, filters_list=self._filters_list)
        
        junction = tf.concat([dense_stem, conv_stem], axis=1)
        
        logits = self._dense(x=junction, units=2)
        return logits
        
    def _make_dense_block(self, x, units):
        x = self._dense(x, units)
        x = tf.nn.relu(x)
        x = self._bn(x, with_dense=True, is_training=self._is_training)
        return x
    
    def _make_conv_block(self, x, filters, downsampling=False):
        x = self._bn(x, with_dense=False, is_training=self._is_training)
        x = tf.nn.relu(x)
        x = self._conv(x, filters, kernel_size=3, stride=1)
        if downsampling:
            x = self._max_pool(x)
        return x
        
    
    def _make_dense_stem(self, x, units_list):
        for units in units_list:
            x = self._make_dense_block(x, units)
        return x
    
    def _make_conv_stem(self, x, filters_list):
        x = self._conv(x, filters_list[0], kernel_size=5, stride=2)
        for idx in range(1, len(filters_list)):
            downsampling = True if (filters_list[idx] > filters_list[idx-1]) else False
            x = self._make_conv_block(x, filters_list[idx])
        x = self._gap(x)
        return x


    

def main():
    from pipeline import get_dataset

    dataset = get_dataset(filenames="../data/tfrecords_format/dataset_wsc_ensemble_12468/test_set_2494.tfrecords")
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

    is_training = tf.placeholder(tf.bool)

    units_list = [10, 20, 10]
    filters_list = [16, 32, 32, 64, 64, 16]
    model = EnsembleModel(
        is_training=is_training,
        units_list=units_list,
        filters_list=filters_list
    )
    logits = model.forward_pass(
        images=batch['image'],
        variables=batch['variables']
    )
    print(logits)

if __name__ == "__main__":
    main()
