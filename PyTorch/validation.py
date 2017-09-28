from __future__ import division
from __future__ import print_function
# torch, torchnet and torchvision
import torch
#
from sklearn.metrics import roc_auc_score
# arson
from arson.metrics import calc_accuracy, ROC
from arson.utils import to_var, to_numpy

class AverageMeter(object):
    """
    Computes and stores the average and current value
    ref. https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(model, data_loader, wsc_loss, bce_loss):
    wsc_loss_meter = AverageMeter()
    bce_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    roc = ROC()

    model.eval()
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx > 5:
            break


        if batch['image'].size(0) < 2:
            continue

        for k in ['image', 'label_weak', 'label']:
            batch[k] = to_var(batch[k])
        output = model(batch['image'])

        # record loss and measure accuracy.
        wsc_loss_value = wsc_loss(output, batch['label_weak'])
        bce_loss_value = bce_loss(output, batch['label'])
        acc_value = calc_accuracy(output, batch['label'])
        roc.append(labels=to_numpy(batch['label']), preds=to_numpy(output))

        n = batch['image'].size(0)
        wsc_loss_meter.update(wsc_loss_value.data[0], n)
        bce_loss_meter.update(bce_loss_value.data[0], n)
        acc_meter.update(acc_value, n)

    roc.compute_roc()

    record = {
        'wsc_loss': wsc_loss_meter.avg,
        'bce_loss': bce_loss_meter.avg,
        'acc': acc_meter.avg,
        'auc': roc.auc,
    }

    return record
    
