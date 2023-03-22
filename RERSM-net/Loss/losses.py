from __future__ import print_function, division
import torch
import warnings
import numpy as np
import torch.nn as nn
warnings.filterwarnings("ignore")


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x, y):
        b, h, w = x.size()
        # loss = torch.sum(torch.abs(x - y)**2)/(h*w)
        loss = torch.sum(torch.abs(x - y)**2)
        return loss
