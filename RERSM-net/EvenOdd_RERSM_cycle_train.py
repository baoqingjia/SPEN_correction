from __future__ import print_function, division
import os
import time
import math
import torch
import argparse
import warnings
import numpy as np
from os.path import join
from scipy.io import savemat
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from Util.utils import *
from Loss.losses import *
from Model.model import *
from Dataset.dataset import *
warnings.filterwarnings("ignore")


GPU = 1
torch.cuda.set_device(GPU)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=150, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--learning_rate', type=int, default=0.0005, help='initial learning rate')
    parser.add_argument("--start", type=int, default=0, help='start number of data')
    parser.add_argument("--end", type=int, default=900, help='end number of data')
    args = parser.parse_args()
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    lr = args.learning_rate
    start = args.start
    end = args.end
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    model_dir = './Pth'
    model_name = 'EvenOdd_RERSM_cycle'

    print("model_name: " + model_name)
    save_dir = join(model_dir, '%s' % model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model = RERSM_Model()
    parameters(model)
    criterion = MSE()
    if cuda:
        model = model.cuda()
        criterion.cuda()
    cudnn.benchmark = False

    print('Loading data ...')
    data_dir = './Dataset/data/simulation_linearMap.mat'
    DisAll, GoodAll, AWhole, AFinal = dataset(data_dir, start, end, 'train')

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, verbose=True, min_lr=0.000001, patience=3)

    print('Training ...')
    epochAll, lossAll = [], []
    for epoch in range(num_epoch):
        t_start = time.time()
        train_err, train_batches = 0, 0
        for Dis, Good in miniB(DisAll, GoodAll, batch_size):
            Dis = norm(Dis)
            Good = norm(Good)
            DisBack = norm(cMultiply(AWhole, Dis))
            GoodBack = norm(cMultiply(AWhole, Good))
            Even, mcEvenOdd = c2ri(Dis)
            NetInput = Variable(mcEvenOdd.type(Tensor))
            optimizer.zero_grad()

            e_PhaseMap, PhaseMap = model(NetInput, creatBasis(Even))

            EvenCorrected = e_PhaseMap * Even

            Corrected = Dis.clone()
            Corrected[:, 1::2, :] = EvenCorrected
            Corrected = norm(Corrected)
            CorrectedBack = norm(cMultiply(AWhole, Corrected))
            CorrectedBack2Blur = norm(cMultiply(AFinal, CorrectedBack))

            loss = criterion(Corrected, CorrectedBack2Blur)
            loss.backward()
            optimizer.step()
            train_err += float(loss.item())
            train_batches += 1
        train_err /= (train_batches * batch_size)
        scheduler.step(train_err)
        t_end = time.time()
        print(" ")
        print("Epoch {}/{}".format(epoch+1, num_epoch))
        print("TrainingLoss: \t{:.6f}".format(train_err))
        print("CorrectTime:  \t{:.6f}s".format(t_end - t_start))
        lossAll.append(train_err)
        epochAll.append(epoch+1)
        if epoch+1 >= num_epoch-10:
            name = 'epoch_%d.pth' % (epoch+1)
            torch.save(model.state_dict(), join(save_dir, name))
            print('Model parameters saved at %s' % join(os.getcwd(), name))
    saveTrainLoss(lossAll, epochAll, save_dir)
    print('Training over!')
