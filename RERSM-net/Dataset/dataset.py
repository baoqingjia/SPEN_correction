from __future__ import print_function, division
import torch
import warnings
import numpy as np
import scipy.io as sio
warnings.filterwarnings("ignore")


def dataset(data_dir, start, end, type):
    x = sio.loadmat(data_dir)
    if type == 'train':
        indices = np.random.permutation(list(range(0, int(end - start))))
        DisAll = torch.tensor(x['Dis'][start:end, :, :][indices]).cuda()
        GoodAll = torch.tensor(x['Good'][start:end, :, :][indices]).cuda()
    else:
        DisAll = torch.tensor(x['Dis'][start:end, :, :]).cuda()
        GoodAll = torch.tensor(x['Good'][start:end, :, :]).cuda()

    AWhole = torch.tensor(x['AWhole']).cuda()
    AFinal = torch.tensor(x['AFinal']).cuda()
    return DisAll, GoodAll, AWhole, AFinal
