import ROOT
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import itertools



class JetImageDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, input_path, tree_name='jet'):
        self.tfile = ROOT.TFile(input_path, 'READ')
        self.jet = self.tfile.Get(tree_name)

    def __len__(self):
        return self.jet.GetEntries()

    def __getitem__(self, idx):
        self.jet.GetEntry(idx)
        image = np.array(self.jet.image, dtype=np.float32).reshape(3, 33, 33)
        # label = np.array(self.jet.label, dtype=np.int64).reshape(2)

        example = {
            'image': image,
            'label': self.jet.label,
            'nMatchedJets': self.jet.nMatchedJets,
            'pt': self.jet.pt,
            'eta': self.jet.eta
        }
        
        return example


class WSCImageDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, input_path, tree_name='jet'):
        self.root_file = ROOT.TFile(input_path, 'READ')
        key = self.root_file.GetListOfKeys().At(0).GetName()
        self.tree = self.root_file.Get(key)

    def __len__(self):
        return int(self.tree.GetEntries())

    def __getitem__(self, idx):
        self.tree.GetEntry(idx)
        image = np.array(self.tree.image, dtype=np.float32).reshape(3, 33, 33)
        label = np.array(self.tree.label, dtype=np.float32).reshape(2)

        example = {
            'image': image,
            'label_weak': np.float32(self.tree.label_weak),
            'label': label,
            'nMatchedJets': self.tree.nMatchedJets,
            'pt': self.tree.pt,
            'eta': self.tree.eta,
            'partonId': self.tree.partonId,
        }
        
        return example



class WSCVarsDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, input_path, tree_name='jet'):
        self.tfile = ROOT.TFile(input_path, 'READ')
        self.jet = self.tfile.Get(tree_name)

    def __len__(self):
        return int(self.jet.GetEntries())

    def __getitem__(self, idx):
        self.jet.GetEntry(idx)

        data = np.array(self.jet.data, dtype=np.float32)

        target = 1 if ( self.jet.partonId == 21 ) else 0

        example = {
            # the discriminating variables
            'data': data,
            # label
            'target_wsc': np.float32(self.jet.label),
            'target': target,
            # other info
            'nMatchedJets': self.jet.nMatchedJets,
            'pt': self.jet.pt,
            'eta': self.jet.eta,
            'partonId': self.jet.partonId,
        }
        
        return example


def show_jet_image(image, image_name, image_format='CHW'):
    if image_format == 'HWC':
        image = np.transpose(image, axes=[2, 0, 1])
        
    # scaling
    for c in range(image.shape[0]):
        image[c] /= image.max()
    
    image = np.transpose(image, axes=[1, 2, 0])
    plt.imshow(image)
    plt.savefig(image_name)
    plt.close()


if __name__ == "__main__":
    path='$LAB/ptJet/data/dataset_wsc_image_150.root'
    dataset = WSCImageDataset(path)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
    )
    data_iter = itertools.cycle(data_loader)
    batch = data_iter.next()
    print(batch['label'].size()) 
