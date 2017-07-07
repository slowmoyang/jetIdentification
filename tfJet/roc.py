import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class ROC(object):
    def __init__(self):
        self.labels = np.array([])
        self.preds = np.array([]) # predictions

    def append_data(self, labels, preds):
        # self.labels = np.concatenate(self.labels, labels)
        self.labels = np.append(self.labels, labels)
        # self.preds = np.concatenate(self.preds, preds)
        self.preds = np.append(self.preds, preds)

    def eval_roc(self):
        self.fpr, self.tpr, _ = roc_curve(self.labels, self.preds)
        self.fnr = 1 - self.fpr
        self.auc = auc(self.fpr, self.tpr)

    def save_roc(self, save_path):
        self.logs = np.vstack([self.tpr, self.fnr, self.fpr]).T
        lname = os.path.join(save_path, 'roc_log_auc_%.2f.csv' % self.auc)
        np.savetxt(lname, self.logs, delimiter=',', header='tpr, fnr, fpr')

    def plot_roc_curve(self, step, title, save_path, show=False):
        fig = plt.figure()
        plt.plot(self.tpr, self.fnr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % self.auc)
        plt.plot([0,1], [1,1], color='navy', lw=2, linestyle='--')
        plt.plot([1,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.xlabel('Quark Jet Efficiency (TPR)')
        plt.ylabel('Gluon Jet Rejection (FNR)')
        plt.title('%s-%d / ROC curve' % (title, step))
        plt.legend(loc='lower left')
        plt.grid()
        fname = 'run-'  + str(step).zfill(6)  + '.png'
        full_path = os.path.join(save_path, fname)
        plt.savefig(full_path)
        if show:
            plt.show()
