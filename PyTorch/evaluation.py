# compatibility with py3
from __future__ import division
from __future__ import print_function
# torch, torchnet and torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# import ...
import argparse
import itertools
# arson
from arson.pipeline import WSCImageDataset
from arson import models
from arson.losses import WSCLoss_v1 as WSCLoss 
from arson.metrics import calc_accuracy, Meter, ROC, OutputHistogram
from arson.logging import make_log_dir, Logger



def test():
    model = models.__dict__[

    checkpoint = torch.load(
    model.load_state_dict(checkpoint[]   

    # Input pipeline
    test_dataset = WSCImageDataset(args.test_data) 

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    wsc_loss = WSCLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    roc = ROC()
    histo = OutputHistogram()



    model.eval() 
    for batch_idx, batch in enumerate(test_loader):
	# test part..
        for k in ['image', 'label_weak', 'label']:
            batch[k] = to_var(batch[k])

        output = model(batch['image'])
        # Calculate WSC and BCE losses.
        wsc_loss_value = wsc_loss(output, batch['label_weak'])
        bce_loss_value = bce_loss(output, batch['label'])
        acc_value = calc_accuracy(output, batch['label'])
        auc_value = roc_auc_score(
            y_true=to_numpy(batch['label'])[:, 1],
            y_score=to_numpy(output)[:, 1]
        )
	roc.append(
            labels=
            preds=
        )
	histo.append(
            labels=
            preds=
            nMatchedJets=
        )
    roc.finish()
    histo.finish()

