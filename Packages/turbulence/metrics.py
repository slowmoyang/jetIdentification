import os
import ROOT
import numpy as np
from sklearn.metrics import roc_curve, auc
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT.gROOT.SetBatch(True)


def calc_accuracy(logits, labels):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        with tf.name_scope('num_correct'):
            num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.summary.scalar('accuracy', accuracy)
    return accuracy


class OldMeter(object):
    def __init__(self, dpath, title='Jet Discrimination'):
        self.dpath = dpath
        self.title = title

        self.step = np.array([])
        # loss
        self.tr_loss = np.array([])
        self.val_loss = np.array([])
        # acc
        self.tr_acc = np.array([])
        self.val_acc = np.array([])

    def append(self, step, tr_loss, val_loss, tr_acc, val_acc):
        self.step = np.r_[self.step, step]
        self.tr_loss = np.r_[self.tr_loss, tr_loss]
        self.val_loss = np.r_[self.val_loss, val_loss]
        self.tr_acc = np.r_[self.tr_acc, tr_acc]
        self.val_acc = np.r_[self.val_acc, val_acc]

    def save_data(self):
        path = os.path.join(self.dpath, 'acc_n_loss.csv')
        logs = np.vstack([self.step, self.tr_loss, self.val_loss, self.tr_acc, self.val_acc]).T
        np.savetxt(path, logs, delimiter=',', header='step, tr_loss, val_loss, tr_acc, val_acc')

    def plot_acc(self):
        plt.figure(figsize=(8, 6))
        plt.rc("font", size=12)

        tr_filtered = lowess(self.tr_acc, self.step,
                             is_sorted=True, frac=0.075, it=0)
        val_filtered = lowess(self.val_acc, self.step,
                              is_sorted=True, frac=0.075, it=0)

        plt.plot(self.step, self.tr_acc,
                 color='navy', lw=2, alpha=0.2, label='Training')
        plt.plot(self.step, self.val_acc,
                 color='darkorange', lw=2, alpha=0.2, label='Validation')
        # smooth
        plt.plot(tr_filtered[:, 0], tr_filtered[:, 1],
                 color='navy', lw=2, label='Training (smooth)')
        plt.plot(val_filtered[:, 0], val_filtered[:, 1],
                 color='darkorange', lw=2, label='Validation (smooth)')

        plt.ylim([0.0, 1.1])

        plt.xlabel('Step')
        plt.ylabel('Acc.')

        plt.title('%s/Accuracy' % self.title)
        plt.legend(loc='lower right')
        plt.grid()

        path = os.path.join(self.dpath, 'acc.png')
        plt.savefig(path)

        plt.close()

    def plot_loss(self):
        plt.figure(figsize=(8, 6))
        plt.rc("font", size=12)

        tr_filtered = lowess(self.tr_loss, self.step, is_sorted=True, frac=0.075, it=0)
        val_filtered = lowess(self.val_loss, self.step, is_sorted=True, frac=0.075, it=0)

        plt.plot(self.step, self.tr_loss,
                 color='navy', lw=2, alpha=0.2, label='Training')
        plt.plot(self.step, self.val_loss,
                 color='darkorange', lw=2, alpha=0.2, label='Validation')

        # smoothing
        plt.plot(tr_filtered[:, 0], tr_filtered[:, 1],
                 color='navy', lw=2, label='Training (smooth)')
        plt.plot(val_filtered[:, 0], val_filtered[:, 1],
                 color='darkorange', lw=2, label='Validation (smooth)')

        plt.xlabel('Step')
        plt.ylabel('Loss')

        plt.title('%s/Loss' % self.title)
        plt.legend(loc='upper left')
        plt.grid()

        path = os.path.join(self.dpath, 'loss.png')
        plt.savefig(path)

        plt.close()

    def finish(self):
        self.save_data()
        self.plot_acc()
        self.plot_loss()


class Meter(object):
    def __init__(self, data_name_list, dpath):
        for data_name in data_name_list:
            setattr(self, data_name, np.array([]))
            
        self.dpath = dpath
        self.waiting_list = []

    def append(self, data_dict):
        for k in data_dict.keys():
            setattr(self, k, np.r_[getattr(self, k), data_dict[k]])
        
    def save(self):
        raise NotImplementedError("")
        
    def plot(self, data_pair_list, title):
        plt.figure(figsize=(8, 6))
        plt.rc("font", size=12)
        
        for x, y in data_pair_list:
            color = self._color(y)
            plt.plot(getattr(self, x), getattr(self, y),
                     color=color, lw=2, alpha=0.2, label=y)
            
            x_filtered, y_filtered = self._smooth(
                getattr(self, x),getattr(self, y))
            
            plt.plot(x_filtered, y_filtered,
                     color=color, lw=2, label=y+'(lowess)')

        plt.ylim([0.0, 1.1])

        plt.xlabel(x)
        plt.ylabel(y)

        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid()
        #plt.show()
        path = os.path.join(self.dpath, title + ".png")
        plt.savefig(path)
        plt.close()
        
    def prepare(self, data_pair_list, title):
        self.waiting_list.append({"data_pair_list": data_pair_list, "title": title})
        
    def finish(self):
        for waiting in self.waiting_list:
            self.plot(**waiting)
        
        
    def _smooth(self, x, y):
        filtered = lowess(y, x, is_sorted=True, frac=0.075, it=0)
        return filtered[:, 0], filtered[:, 1]
        
    def _color(self, y):
        if 'tr' in y:
            color = 'navy'
        elif 'val' in y:
            color = 'orange'
        else:
            color = np.random.rand(3,1)
        return color


class ROC(object):
    def __init__(self, dpath, step, title):
        self.dpath = dpath
        self.step = step
        self.title = title

        self.labels = np.array([])
        self.preds = np.array([])  # predictions

        # uninitialized attributes
        self.fpr = None
        self.tpr = None
        self.fnr = None
        self.auc = None

    def append(self, labels, preds):
        # self.labels = np.r_[self.labels, labels]
        # self.preds = np.r_[self.preds, preds]
        self.labels = np.append(self.labels, labels)
        self.preds = np.append(self.preds, preds)


    def compute_roc(self):
        self.fpr, self.tpr, _ = roc_curve(self.labels, self.preds)
        self.fnr = 1 - self.fpr
        self.auc = auc(self.fpr, self.tpr)

    def save_roc(self, path):
        logs = np.vstack([self.tpr, self.fnr, self.fpr]).T
        np.savetxt(path, logs, delimiter=',', header='tpr, fnr, fpr')

    def plot_roc_curve(self, path):
        # fig = plt.figure()
        plt.plot(self.tpr, self.fnr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.3f)' % self.auc)
        plt.plot([0, 1], [1, 1], color='navy', lw=2, linestyle='--')
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

        filename_format = 'roc_step-%s_auc-%.3f.%s' % (str(self.step).zfill(6), self.auc, '%s')

        csv_path = os.path.join(self.dpath, filename_format % 'csv')
        plot_path = os.path.join(self.dpath, filename_format % 'png')

        self.save_roc(csv_path)
        self.plot_roc_curve(plot_path)



class OutputHistogram(object):
    '''
      Gluon discriminator
      1 : gluon-like
      0 : quark-like
    '''
    def __init__(self, dpath, step):
        self.step = step
        filename_format = 'histogram_step-%s.%s' % (str(self.step).zfill(6), '%s')
        self.root_path = os.path.join(dpath, filename_format%'root')
        self.plot_path = os.path.join(dpath, filename_format%'png')

        self.quark_all = ROOT.TH1F("Quark(all)", "", 100, 0, 1)
        self.gluon_all = ROOT.TH1F("Gluon(all)","", 100, 0, 1)

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

    def plot(self):
        val = ROOT.TFile(self.root_path, "READ")
        c = ROOT.TCanvas("c2", "", 800, 600)
        all = [val.quark_all, val.gluon_all]
        two = [val.quark_2, val.gluon_2]
        three = [val.quark_3, val.gluon_3]

        for p in all+two+three:
            p.Scale(1.0/p.GetEntries())

        val.quark_all.SetFillColor(46)
        val.gluon_all.SetFillColor(38)

        for p, color in zip(two+three, [6, 7, 2, 4]):
            p.SetLineColor(color)
    
        val.quark_all.SetFillStyle(3352)
        val.gluon_all.SetFillStyle(3325)

        for p in two+three:
            p.SetLineWidth(2)

        val.quark_all.Draw('hist')
        val.gluon_all.Draw('hist SAME')

        for p in (two+three):
            p.Draw('SAME')

        val.quark_all.SetStats(0)

        # val.quark_all.GetYaxis().SetRangeUser(0, 0.3)
        c.BuildLegend( 0.35,  0.20,  0.60,  0.45).SetFillColor(0)
        c.SetLogy()
        c.SetGrid()
        # title, axis,
        ltx = ROOT.TLatex()
        ltx.SetNDC()
        title = '%s step' % self.step
        ltx.DrawLatex(0.40, 0.93, title)
        ltx.SetTextSize(0.025)
        ltx.DrawLatex(0.85, 0.03, 'gluon-like')
        ltx.DrawLatex(0.07, 0.03, 'quark-like')
        c.Draw()
        c.SaveAs(self.plot_path)
        c.Close()

    def finish(self):
        # self.draw()
        self.save()
        self.plot()
