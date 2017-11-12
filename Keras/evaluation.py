from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import argparse
from datetime import datetime
from tqdm import tqdm
#import numpy as np
import matplotlib as mpl
mpl.use('Agg')

#import tensorflow as tf

import keras
#from keras import (
#    optimizers,
#    losses,
#    metrics)
from keras.models import load_model
from keras.utils import multi_gpu_model 

#import keras.backend as K

from models import add_an_sigmoid_layer
from pipeline import make_data_loader

#from custom_losses import binary_cross_entropy_with_logits
#from custom_metrics import accuracy_with_logits
from meters import ROCMeter, OutHist
from utils import (
    get_log_dir,
    get_saved_model_paths
)



def evaluate(saved_model_path,
             step,
             train_data,
             test_dijet_data,
             test_zjet_data,
             log_dir):
    # TEST

    model_logit = load_model(saved_model_path)
    model_sigmoid = add_an_sigmoid_layer(model_logit)

    model = multi_gpu_model(model_sigmoid, 2)

    out_hist = OutHist(
        dpath=log_dir.output_histogram.path,
        step=step,
        dname_list=["train", "test_dijet", "test_zjet"])

    # on training data
    train_data_loader = make_data_loader(
        path=train_data,
        batch_size=1000,
        cyclic=False)

    for x, y in train_data_loader:
        preds = model.predict_on_batch(x)
        out_hist.fill(dname="train", labels=y, preds=preds)

    # Test on dijet dataset
    test_dijet_loader = make_data_loader(
        path=test_dijet_data,
        batch_size=1000,
        cyclic=False)

    roc_dijet = ROCMeter(
        dpath=log_dir.roc.path,
        step=step,
        title="Test on Dijet",
        prefix="dijet_"
    )

    for x, y in test_dijet_loader:
        preds = model.predict_on_batch(x)

        roc_dijet.append(labels=y, preds=preds)
        out_hist.fill(dname="test_dijet", labels=y, preds=preds)

    roc_dijet.finish()

    # Test on Z+jet dataset
    test_zjet_loader = make_data_loader(
        path=test_zjet_data,
        batch_size=1000,
        cyclic=False)

    roc_zjet = ROCMeter(
        dpath=log_dir.roc.path,
        step=step,
        title="Test on Z+jet",
        prefix="zjet_"
    )

    for x, y in test_zjet_loader:
        preds = model.predict_on_batch(x)
        roc_zjet.append(labels=y, preds=preds)
        out_hist.fill("test_zjet", labels=y, preds=preds)

    roc_zjet.finish()

    out_hist.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, required=True)

    parser.add_argument(
	    '--log_dir', type=str, required=True,
    	help='the directory path of dataset')

    args = parser.parse_args()

    if args.train_data.lower() == "dijet":
        train_data = "../data/FastSim/dijet/training_dijet_466554_prep.root"
        test_dijet_data = "../data/FastSim/dijet/test_dijet_after_dijet_186880_prep.root"
        test_zjet_data = "../data/FastSim/dijet/test_zjet_after_dijet_176773_prep.root"
    elif args.train_data.lower() == "zjet":
        train_data = "../data/FastSim/zjet/training_zjet_440168_prep.root"
        test_dijet_data = "../data/FastSim/zjet/test_dijet_after_zjet_186880_prep.root"
        test_zjet_data = "../data/FastSim/zjet/test_zjet_after_zjet_176773_prep.root"
    else:
        raise ValueError("")

    log_dir = get_log_dir(path=args.log_dir, creation=False)


    path_and_step = get_saved_model_paths(log_dir.saved_models.path)
    for i, (saved_model_path, step) in enumerate(path_and_step):
        print("\n\n\n[{i}/{total}]: {path}".format(
            i=i, total=len(path_and_step), path=saved_model_path))
        evaluate(
            saved_model_path,
            step,
            train_data,
            test_dijet_data,
            test_zjet_data,
            log_dir)


if __name__ == "__main__":
    main()
