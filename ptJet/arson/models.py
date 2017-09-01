import torch
import torch.nn as nn
import torch.nn.functional as F
# custom
from utils import calc_feature_map_size as calc_fms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # x: [B, 3, 33, 33]
        self.layer1, fms = self.make_conv_layer(cin=3, cout=64, first=True) # [B, 64, 16, 16]
        self.layer2, fms = self.make_conv_layer(cin=64, cout=128, fms=fms) # [B, 128, 9, 9]
        self.layer3, fms = self.make_conv_layer(cin=128, cout=256, fms=fms) # [B, 256, 5, 5]
        self.layer4, fms = self.make_conv_layer(cin=256, cout=512, fms=fms) # [B, 512, 3, 3]
              
        # flatten: [B, C*H*W]=[B, 256*8*8]
        # fc: []
        self.fc = nn.Linear(512*fms*fms, 2)
        self.softmax = nn.Softmax()

    def make_conv_layer(self,cin, cout, fms=33, first=False):
        k1 = 5 if first else 3
        layer = nn.Sequential(
            # nn.BatchNorm2d(cin),
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
            nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=k1, padding=2),
            nn.ReLU(),
            # nn.BatchNorm2d(cout),
            nn.Conv2d(in_channels=cout, out_channels=cout, kernel_size=3, padding=2),
            nn.ReLU(),
            # MaxPool2d(k, s=None, p=0)
            # If s=None, s=k
            nn.MaxPool2d(kernel_size=2),
        )
        
	fms = calc_fms(conv=True, Lin=fms, k=k1)
	fms = calc_fms(conv=True, Lin=fms, k=3)
	fms = calc_fms(conv=False, Lin=fms, k=2)
        return layer, fms

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out


class DNN(nn.Module):
    def __init__(self, in_features):
        super(DNN, self).__init__()
        # fc: []
        self.fc1 = self.make_layer(in_fueatres, 20)
        self.fc2 = self.make_layer(20, 100)
        self.fc3 = self.make_layer(100, 200)
        self.fc4 = self.make_layer(200, 100)
        self.fc5 = self.make_layer(100, 20)
        self.logits = nn.Linear(20, 2)

    def make_layer(self, in_features, out_features):
        layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        )
        return layer

        
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc(out)
        return out
