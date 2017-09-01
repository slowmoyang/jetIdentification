# compatibility with py3
from __future__ import division
from __future__ import print_function
# torch, torchnet and torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
# other libraries
import os
import ROOT
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# arson
from utils import to_numpy


ROOT.gROOT.SetBatch(True)


def calc_accuracy(output, label):

    # argmax
    _, io = output.max(1)
    _, il = label.max(1)
    correct = io.eq(il)
    correct = correct.sum()
    correct = to_numpy(correct)
    correct = float(correct)
    total = io.size(0)
    return correct / total


class Accuracy(nn.modules.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
    def forward(self, output, target):
        output_mean = torch.mean(output[:, 1])
        target_mean = torch.mean(target)
        diff = output_mean - target_mean
        loss = torch.abs(diff)
        return loss


class Meter(object):
    def __init__(self, dpath):
        # loss, accuracy
        path = os.path.join(dpath, 'metric.root')
        self.root_file = ROOT.TFile(path, "RECREATE")
        self.tree = ROOT.TTree('log', 'log')

        # Buffer
        self.step = np.zeros(shape=[1], dtype=np.int32)
        # trainging
        self.tr_wsc_loss = np.zeros(shape=[1], dtype=np.float32)
        self.tr_bce_loss = np.zeros(shape=[1], dtype=np.float32)
        self.tr_acc = np.zeros(shape=[1], dtype=np.float32)
        self.tr_auc = np.zeros(shape=[1], dtype=np.float32)
        # validation
        self.val_wsc_loss = np.zeros(shape=[1], dtype=np.float32)
        self.val_bce_loss = np.zeros(shape=[1], dtype=np.float32)
        self.val_acc = np.zeros(shape=[1], dtype=np.float32)
        self.val_auc = np.zeros(shape=[1], dtype=np.float32)

        # Branch
        self.tree.Branch("step", self.step, "step/I")
        # training
        self.tree.Branch("tr_wsc_loss", self.tr_wsc_loss, "tr_wsc_loss/F")
        self.tree.Branch("tr_bce_loss", self.tr_bce_loss, "tr_bce_loss/F")
        self.tree.Branch("tr_acc",      self.tr_acc,         "tr_acc/F")
        self.tree.Branch("tr_auc",      self.tr_auc,         "tr_auc/F")
        # validation
        self.tree.Branch("val_wsc_loss", self.val_wsc_loss, "val_wsc_loss/F")
        self.tree.Branch("val_bce_loss", self.val_bce_loss, "val_bce_loss/F")
        self.tree.Branch("val_acc",      self.val_acc,      "val_acc/F")
        self.tree.Branch("val_auc",      self.val_auc,      "val_auc/F")

    def fill(self, branch_dict):
        self.step[0] = branch_dict['step']
        # training
        self.tr_wsc_loss[0] = branch_dict['tr_wsc_loss']
        self.tr_bce_loss[0] = branch_dict['tr_bce_loss']
        self.tr_acc[0] = branch_dict['tr_acc']
        self.tr_auc[0] = branch_dict['tr_auc']
        # validation
        self.val_wsc_loss[0] = branch_dict['val_wsc_loss']
        self.val_bce_loss[0] = branch_dict['val_bce_loss']
        self.val_acc[0] = branch_dict['val_acc']
        self.val_auc[0] = branch_dict['val_auc']

        self.tree.Fill()

    def finish(self):
        self.tree.Write()
        self.root_file.Close()


class ROC(object):
    def __init__(self, step, title, dpath):
        self.step = step
        self.dpath = dpath
        self.title = title
        fname_format = 'step-%s_auc-%s.%s' % (str(step).zfill(6), '%s', '%s')
        self.path_format = os.path.join(dpath, fname_format)
        self.labels = np.array([])
        self.preds = np.array([])  # predictions

    def append(self, labels, preds):
        self.labels = np.append(self.labels, labels)
        self.preds = np.append(self.preds, preds)

    def compute_roc(self):
        self.fpr, self.tpr, _ = roc_curve(self.labels, self.preds)
        self.fnr = 1 - self.fpr
        self.auc = auc(self.fpr, self.tpr)

    def save_roc(self, path):
        tf = ROOT.TFile(path,'RECREATE')
        tree = ROOT.TTree('roc', 'roc')
        tpr_buffer = np.zeros(shape=[1], dtype=np.float32)
        fnr_buffer = np.zeros(shape=[1], dtype=np.float32)
        tree.Branch('fpr')
        tree.Branch('fnr')
        for i in range(len(self.tpr)):
            tpr_buffer[0] = self.tpr[i]
            fnr_buffer[0] = self.fnr[i]
            tree.Fill() 
        tree.Write()
        tf.Close()

    def plot_roc_curve(self, path):
        # fig = plt.figure()
        plt.plot(self.tpr, self.fnr, color='darkorange',
                 lw=3, label='ROC curve (area = %0.3f)' % self.auc)
        plt.plot(s[0, 1], [1, 1], color='navy', lw=2, linestyle='--')
        plt.plot([1, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.xlabel('Quark Jet Efficiency (TPR)')
        plt.ylabel('Gluon Jet Rejection (FNR)')
        plt.title('%s-%d / ROC curve' % (self.title, self.step))
        plt.legend(loc='lower left')
        plt.grid()
        plt.savefig(path)
        plt.close()

    def finish(self):
        self.compute_roc()
        root_path = self.path_format % (str(self.ac), 'root')
        self.save_roc(path=root_path)
        plot_path = self.path_format % (str(self.ac), 'png')
        self.plot_roc_curve(path=plot_path)


class OutputHistogram(object):
    def __init__(self, dpath, step):
        fname = 'histogram_%s.root' % str(step).zfill(6)
        self.root_path = os.path.join(dpath, fname)

        self.quark_all = ROOT.TH1F("Quark(all)","", 100, 0, 1)
        self.gluon_all = ROOT.TH1F("Gluon(all)", "", 100, 0, 1)

        self.quark_2 = ROOT.TH1F("Quark(nMatJets=2)", "", 100, 0, 1)
        self.gluon_2 = ROOT.TH1F("Gluon(nMatJets=2)", "", 100, 0, 1)
     
        self.quark_3 = ROOT.TH1F("Quark(nMatJets=3)", "", 100, 0, 1)
        self.gluon_3 = ROOT.TH1F("Gluon(nMatJets=3)", "", 100, 0, 1)

    def fill(self, labels, preds, nMatchedJets):
        for n, i in enumerate(labels.argmax(axis=1)):
            gluon_likeness = preds[n, 1]
            # gluon
            if i:
                # all
                self.gluon_all.Fill(gluon_likeness)
                if nMatchedJets[n] == 2:
                    self.gluon_2.Fill(gluon_likeness)
                elif nMatchedJets[n] == 3:
                    self.gluon_3.Fill(gluon_likeness)
            # quark
            else:
                # all
                self.quark_all.Fill(gluon_likeness)
                if nMatchedJets[n] == 2:
                    self.quark_2.Fill(gluon_likeness)
                elif nMatchedJets[n] == 3:
                    self.quark_3.Fill(gluon_likeness)

    def save(self):
        writer = ROOT.TFile(self.root_path, 'RECREATE')
        # Overall
        self.quark_all.Write('quark_all')  
        self.gluon_all.Write('gluon_all')
        # nMatachedJets == 2
        self.quark_2.Write('quark_2')  
        self.gluon_2.Write('gluon_2')
        # nMatchedJets == 3
        self.quark_3.Write('quark_3')  
        self.gluon_3.Write('gluon_3')
        writer.Close()

    def finish(self):
        # self.draw()
        self.save()
