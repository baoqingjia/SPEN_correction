from __future__ import print_function, division
import warnings
import torch.nn as nn
from Util.utils import creatMap
from Model.network import ResEncoder
warnings.filterwarnings("ignore")


class RERSM_Model(nn.Module):
    def __init__(self):
        super(RERSM_Model, self).__init__()
        self.net = ResEncoder(4, 1)

    def forward(self, x, basis):
        b, c, h, w = x.size()
        y = self.net(x)
        feature = []
        feature.append(y)
        e_PhaseMap, PhaseMap = creatMap(b, int(h / 2), w, basis, feature)
        return e_PhaseMap, PhaseMap
