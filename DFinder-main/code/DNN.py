import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F


class DNN_for_feature_extra(nn.Module):
    def __init__(self):
        super(DNN_for_feature_extra, self).__init__()
        self.dense1 = torch.nn.Linear(2159,1024)
        self.dense2 = torch.nn.Linear(1024,512)
        self.dense3 = torch.nn.Linear(512,256)
        self.dense4 = torch.nn.Linear(256,64)

    def forward(self,feat):
        feat1 = F.relu(self.dense1(feat))
        feat2 = F.relu(self.dense2(feat1))
        feat3 = F.relu(self.dense3(feat2))
        feat4 = self.dense4(feat3)
        return feat4