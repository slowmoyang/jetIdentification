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
from arson.metrics import calc_accuracy, Meter, ROC
from arson.logging import make_log_dir, Logger


def save_checkpoint(state, filename, is_best):
    torch.save(state, filename)
    if is_best:




def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default='./data/dataset_wsc_image_1.root')
    parser.add_argument("--validation_data", type=str, default='./data/dataset_wsc_image_1.root')
    parser.add_argument("--test_data", type=str, default='./data/dataset_wsc_image_1.root')
    parser.add_argument("--log_dir", type=str, default='./logs/wsc-01')
    #
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=500)
    # Hyperparameter
    args = parser.parse_args()

    logdir = make_log_dir(path=args.logdir)

    # Input pipeline
    train_dataset = WSCImageDataset(args.training_data) 
    val_dataset = WSCImageDataset(args.val_data) 

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_iter = itertools.cycle(val_loader)

    # Define model, criterion and optimization algorithm
    model = models.__dict__[args.model]()
    model.cuda()
    # Define weakly supervised classification loss and use it as criterion.
    wsc_loss = WSCLoss()
    # Define binary cross entropy with logits and use it as metric
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr):

    # Training
    step = 0
    model.train()
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            step += 1
            # training
 	    batch['image'], batch['target_wsc']  = batch['image'].cuda(), batch['target_wsc'].cuda()
	    batch['image'] = Variable(batch['image'])
	    optimizer.zero_grad()
	    output = model(batch['image'])
	    wsc_loss_value = wsc_loss(output, batch['target_wsc'])
	    wsc_loss_value.backward()
	    optimizer.step()
           
            # validation
            if(step % args.val_freq == 0):
                # switch to evaluation mode
                model.eval()
                # validation data batch
                batch = val_iter.next()
 	        batch['image'], batch['target_wsc']  = batch['image'].cuda(), batch['target_wsc'].cuda()
	        batch['image'] = Variable(batch['image'])
                #
                output = model(batch['image'])
                # Calculate WSC and BCE losses.
                wsc_loss_value = wsc_loss(batch['image'], batch['target_wsc'])
                bce_loss_value = bce_loss(batch['image'], batch['target'])
                acc_value = 
                # Meter
                ''' :D  '''
                # print..
                print('Epochs: [%d/%d], Iter: [%d/%d]'
                      % ((epoch+1), args.num_epochs, (batch_idx+1), len(train_data)//args.batch_size)
                print('WSC Loss: %.4f, BCE Loss: %.4f'
                      % (wsc_loss_value.data[0], bce_loss_value.data[0])
                print('ACC.: %.4f, AUC: %.4f'
                      % (acc.data[0], ###)

                # switch to evaluation mode
                model.train()             

            state_dict = {
                'mdoel': model.state_dict(),
                'step': step,
                ''
            }
            if(step % args.save_freq == 0):
                # model_step_acc_auc.pth.gz
                filename_format = '%s_%d_acc-%.3f_auc-%.3f.pth.gz'

                save_checkpoint()


def test():
    # load best model..
    best_model = ...

    # Input pipeline
    train_dataset = WSCImageDataset(args.training_data) 

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )




    roc = ROC()
    histo = OutputHistogram()
    for batch_idx, batch in enumerate(test_loader):
	# test part..

	roc.append(
	histo.append(..



def main():

