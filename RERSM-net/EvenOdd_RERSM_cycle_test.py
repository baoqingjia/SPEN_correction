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
from skimage.metrics import structural_similarity as compare_ssim
warnings.filterwarnings("ignore")


GPU = 1
torch.cuda.set_device(GPU)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument("--start", type=int, default=900, help='start number of data')
    parser.add_argument("--end", type=int, default=1000, help='end number of data')
    args = parser.parse_args()
    batch_size = args.batch_size
    start = args.start
    end = args.end
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    model_dir = './Pth'
    model_name = 'EvenOdd_RERSM_cycle'

    print("model_name: " + model_name)
    pth_dir = join(model_dir, '%s' % model_name)
    save_dir = pth_dir + '/testResult'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model = RERSM_Model()
    parameters(model)
    if cuda:
        model = model.cuda()
    cudnn.benchmark = False

    model.load_state_dict(torch.load(pth_dir + '/epoch_150.pth', map_location='cuda:{}'.format(GPU)))

    print('Loading data ...')
    data_dir = './Dataset/data/simulation_linearMap.mat'
    DisAll, GoodAll, AWhole, AFinal = dataset(data_dir, start, end, 'test')

    print('Testing ...')
    test_batches, test_psnr, test_ssim, t_sub = 0, 0, 0, 0
    CorrectedBackAll, GoodBackAll, DisBackAll, ErrorAll, PhaseMapAll, PSNR, SSIM = [], [], [], [], [], [], []
    for Dis, Good in miniB(DisAll, GoodAll, batch_size):
        Dis = norm(Dis)
        Good = norm(Good)
        DisBack = norm(cMultiply(AWhole, Dis)).data.cpu().numpy()
        GoodBack = norm(cMultiply(AWhole, Good)).data.cpu().numpy()
        Even, mcEvenOdd = c2ri(Dis)
        NetInput = Variable(mcEvenOdd.type(Tensor))

        t_start = time.time()
        with torch.no_grad():
            e_PhaseMap, PhaseMap = model(NetInput, creatBasis(Even))
        t_end = time.time()
        t_sub += (t_end - t_start)

        PhaseMap = PhaseMap.data.cpu().numpy()
        EvenCorrected = e_PhaseMap * Even

        Corrected = Dis.clone()
        Corrected[:, 1::2, :] = EvenCorrected
        CorrectedBack = norm(cMultiply(AWhole, Corrected)).data.cpu().numpy()

        Error = GoodBack - CorrectedBack

        for i in range(batch_size):
            single_psnr = complex_psnr(CorrectedBack[i, :, :], GoodBack[i, :, :])
            PSNR.append(single_psnr)
            test_psnr += single_psnr
            single_ssim = compare_ssim(np.abs(CorrectedBack[i, :, :]), np.abs(GoodBack[i, :, :]), multichannel=False)
            SSIM.append(single_ssim)
            test_ssim += single_ssim

            PhaseMapAll.append(PhaseMap[i, :, :])
            CorrectedBackAll.append(CorrectedBack[i, :, :])
            GoodBackAll.append(GoodBack[i, :, :])
            ErrorAll.append(Error[i, :, :])
            DisBackAll.append(DisBack[i, :, :])
        del Good, Dis, Even, mcEvenOdd, EvenCorrected, Corrected, CorrectedBack, GoodBack, Error, DisBack, e_PhaseMap, PhaseMap
        test_batches += 1
    t = t_sub / (test_batches * batch_size)
    test_psnr /= (test_batches * batch_size)
    test_ssim /= (test_batches * batch_size)
    print(" ")
    print("Time_slice: {:.4f} ms".format(t * 1000))
    print("Test_PSNR: {:.4f}".format(test_psnr))
    print("Test_SSIM: {:.4f}".format(test_ssim))
    del DisAll, GoodAll, AWhole, AFinal

    print(" ")
    print("saving figures")
    i = 1
    for correctedBack, goodBack, error, disBack, phaseMap in zip(CorrectedBackAll, GoodBackAll, ErrorAll, DisBackAll, PhaseMapAll):
        figure = np.concatenate([np.abs(goodBack), np.abs(correctedBack), np.abs(disBack)], 1)
        plt.imsave(join(save_dir, 'figure{}.png'.format(i)), figure, cmap='gray')
        plt.imsave(join(save_dir, 'correctedBack{}.png'.format(i)), np.abs(correctedBack), cmap='gray')
        plt.imsave(join(save_dir, 'goodBack{}.png'.format(i)), np.abs(goodBack), cmap='gray')
        plt.imsave(join(save_dir, 'error{}.png'.format(i)), np.abs(error), cmap='jet')
        plt.imsave(join(save_dir, 'disBack{}.png'.format(i)), np.abs(disBack), cmap='gray')
        plt.imsave(join(save_dir, 'phaseMap{}.png'.format(i)), np.abs(phaseMap), cmap='jet')
        i += 1
    del correctedBack, goodBack, error, disBack, phaseMap
    savemat(join(save_dir, 'PhaseMapAll.mat'), {'PhaseMapAll': PhaseMapAll})
    savemat(join(save_dir, 'CorrectedBackAll.mat'), {'CorrectedBackAll': CorrectedBackAll})
    savemat(join(save_dir, 'GoodBackAll.mat'), {'GoodBackAll': GoodBackAll})
    savemat(join(save_dir, 'ErrorAll.mat'), {'ErrorAll': ErrorAll})
    savemat(join(save_dir, 'DisBackAll.mat'), {'DisBackAll': DisBackAll})
    savePSNRSSIM(PSNR, SSIM, save_dir)
    print('Testing over!')
