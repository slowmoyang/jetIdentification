import os
import argparse
import ROOT
from utils import get_log_dir


class QGHistogram(object):
    '''
      Gluon discriminator
      1 : gluon-like
      0 : quark-like
    '''
    def __init__(self, dpath, step, is_training_data):
        fname = 'histogram_%s.root' % str(step).zfill(6)
        which_data_set = 'training' if is_training_data else 'validation'
        self.root_path = os.path.join(dpath, fname)

        self.quark_all = ROOT.TH1F("Quark(all/%s)" % which_data_set,"", 100, 0, 1)
        self.gluon_all = ROOT.TH1F("Gluon(all/%s)" % which_data_set,"", 100, 0, 1)

        self.quark_2 = ROOT.TH1F("Quark(nMatJets=2/%s)" % which_data_set,"", 100, 0, 1)
        self.gluon_2 = ROOT.TH1F("Gluon(nMatJets=2%s)" % which_data_set,"", 100, 0, 1)
     
        self.quark_3 = ROOT.TH1F("Quark(nMatJets=3/%s)" % which_data_set,"", 100, 0, 1)
        self.gluon_3 = ROOT.TH1F("Gluon(nMatJets=3/%s)" % which_data_set,"", 100, 0, 1)

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
       

def draw_qg_histogram(tr_path, val_path, step, path):
    val = ROOT.TFile(val_path, "READ")

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
    title = '%d step' % step
    ltx.DrawLatex(0.40, 0.93, title)
    ltx.SetTextSize(0.025)
    ltx.DrawLatex(0.85, 0.03, 'gluon-like')
    ltx.DrawLatex(0.07, 0.03, 'quark-like')
    c.Draw()
    c.SaveAs(path)
    c.Close()


def parse_qg_histogram_fname(path):
    fname = os.path.split(path)[-1]
    without_extension = os.path.splitext(fname)[0]
    step = int(without_extension.split('_')[-1])
    return step
    

def draw_all_qg_histograms(qg_histogram_dir):
    tr_fname_list = os.listdir(qg_histogram_dir.training.path)
    val_fname_list = os.listdir(qg_histogram_dir.validation.path)
    tr_path_list = map(lambda fname: os.path.join(qg_histogram_dir.training.path, fname), tr_fname_list)
    val_path_list = map(lambda fname: os.path.join(qg_histogram_dir.validation.path, fname), val_fname_list)
    tr_path_list.sort()
    val_path_list.sort()

    for tr_path, val_path in zip(tr_path_list, val_path_list):
        step = parse_qg_histogram_fname(tr_path)
        if step != parse_qg_histogram_fname(val_path):
            raise ValueError(':p')
        fname = 'step_%d.png' % step
        output_path = os.path.join(qg_histogram_dir.path, fname)
        draw_qg_histogram(tr_path, val_path, step, output_path)

def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dname", default='test_04')

    args = parser.parse_args()

    log_dir = get_log_dir(dname=args.dname, creation=False)

    draw_all_qg_histograms(qg_histogram_dir=log_dir.qg_histogram)
 

if __name__ == '__main__':
    main()
