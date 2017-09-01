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
import os
import argparse
import itertools
from sklearn.metrics import roc_auc_score
# arson
from arson.pipeline import WSCImageDataset
from arson import models
from arson.losses import WSCLoss_v1 as WSCLoss 
from arson.metrics import calc_accuracy, Meter, ROC
from arson.logging import make_log_dir, Logger
from arson.utils import to_var, to_numpy, load_datasets
#
from validation import validate

def train():
    parser = argparse.ArgumentParser()
    # Env
    parser.add_argument("--datasets_dir", type=str, default='./data/dataset_wsc_image_13503/')
    parser.add_argument("--model", type=str, default='CNN')
    parser.add_argument("--log_dir", type=str, default='test-01')
    #
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=50)
    # optimizer
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=0.0015)
    args = parser.parse_args()

    log_dir = make_log_dir(dname=args.log_dir)

    # Input pipeline
    datasets = load_datasets(args.datasets_dir)
    training_dataset = WSCImageDataset(datasets['training'])
    validation_dataset = WSCImageDataset(datasets['validation']) 

    train_loader = torch.utils.data.DataLoader(
        dataset=training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=500,
        shuffle=True,
    )

    # Define model
    model = models.__dict__[args.model]()
    model.cuda()
    # Define weakly supervised classification loss and use it as the criterion.
    wsc_loss = WSCLoss()
    # Define binary cross entropy with logits and use it as the metric
    bce_loss = nn.BCEWithLogitsLoss()
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.__dict__[args.otpim](model.parameters())

    meter = Meter(dpath=log_dir.path)

    # Training
    step = 0
    model.train()
    for epoch in range(args.num_epochs):
        print("Epoch: %d" % epoch)
        for batch_idx, batch in enumerate(train_loader):

            # Tensor -> CUDA Tensor -> Variable
            batch['image'] = to_var(batch['image'])
            batch['label_weak'] = to_var(batch['label_weak'])

            # Compute output and loss
	    output = model(batch['image'])
	    wsc_loss_value = wsc_loss(output, batch['label_weak'])

            # compute gradient and do Adam step
	    optimizer.zero_grad()
	    wsc_loss_value.backward()
	    optimizer.step()


            # validation
            if(step % args.val_freq == 0):
                batch['label'] = to_var(batch['label'])

                bce_loss_value = bce_loss(output, batch['label'])
                acc_value = calc_accuracy(output, batch['label'])
                auc_value = roc_auc_score(
                    y_true=to_numpy(batch['label'][:, 1]),
                    y_score=to_numpy(output[:, 1]),
                )

                branch_dict = {
                    'step': step,
                    'tr_wsc_loss': wsc_loss_value.data[0],
                    'tr_bce_loss': bce_loss_value.data[0],
                    'tr_acc': acc_value,
                    'tr_auc': auc_value
                }


                # validation data batch
                record = validate(
                    model=model, data_loader=val_loader,
                    wsc_loss=wsc_loss, bce_loss=bce_loss
                )

                branch_dict['val_wsc_loss'] = record['wsc_loss']
                branch_dict['val_bce_loss'] = record['bce_loss']
                branch_dict['val_acc'] = record['acc']
                branch_dict['val_auc'] = record['auc']

                # Meter
                meter.fill(branch_dict)

                # print..
                print('Epochs: [%d/%d], Iter: [%d/%d], Step: %d'
                      % ((epoch+1), args.num_epochs, (batch_idx+1), len(training_dataset)//args.batch_size, (step+1)))
                print('WSC Loss: %.4f, BCE Loss: %.4f'
                      % (wsc_loss_value.data[0], bce_loss_value.data[0]))
                print('ACC.: %.4f, AUC: %.4f\n'
                      % (acc_value, auc_value))

                # switch to evaluation mode
                model.train()             

                state={
                    'step': step+1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                checkpoint_name = "checkpoint_step-%s_acc-%.4f_auc-%.4f.pth.tar" % \
                    (str(step+1).zfill(6), record['acc'], record['auc'])
                checkpoint_path = os.path.join(log_dir.checkpoint.path, checkpoint_name)
                torch.save(obj=state, f=checkpoint_path)
            step += 1
    meter.finish()

if __name__ == "__main__":
    train()
