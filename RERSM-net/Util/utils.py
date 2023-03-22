from __future__ import print_function, division
import math
import torch
import warnings
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def norm(x):
    y = x / torch.max(torch.abs(x))
    return y


def complex_psnr(x, y):
    mse = np.mean(np.abs(x - y)**2)
    psnr = 10 * np.log10(1./mse)
    return psnr


def savePSNRSSIM(PSNR, SSIM, save_dir):
    np.savetxt(join(save_dir, r'psnr.txt'), np.array(PSNR), fmt='%.4f', delimiter='\n')
    np.savetxt(join(save_dir, r'ssim.txt'), np.array(SSIM), fmt='%.4f', delimiter='\n')


def saveTrainLoss(lossAll, epochAll, save_dir):
    np.savetxt(join(save_dir, r'train_loss.txt'), np.array(lossAll), fmt='%.6f', delimiter='\n')
    plt.figure()
    plt.plot(epochAll, lossAll)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(join(save_dir, 'train_loss.png'))


def c2ri(x):
    b, h, w = x.size()
    Odd = x[:, 0::2, :]
    Even = x[:, 1::2, :]
    mcEvenOdd = torch.zeros(b, 4, int(h / 2), w).cuda()
    mcEvenOdd[:, 0, :, :] = Even.real
    mcEvenOdd[:, 1, :, :] = Even.imag
    mcEvenOdd[:, 2, :, :] = Odd.real
    mcEvenOdd[:, 3, :, :] = Odd.imag
    mcEvenOdd = np.pad(mcEvenOdd.data.cpu().numpy(), ((0, 0), (0, 0), (int(h/4), int(h/4)), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    mcEvenOdd = torch.tensor(mcEvenOdd).cuda()
    return Even, mcEvenOdd


def miniB(x1, x2, batch):
    for i in range(0, len(x1), batch):
        yield x1[i: i+batch], x2[i: i+batch]


def cMultiply(a, x):
    b, h, w = x.size()
    y = torch.complex(torch.zeros((b, h, w), dtype=torch.float64), torch.zeros((b, h, w), dtype=torch.float64)).cuda()
    for i in range(b):
        x1 = x[i, :, :]
        y[i, :, :] = torch.view_as_complex(torch.stack((a.real @ x1.real - a.imag @ x1.imag, a.real @ x1.imag + a.imag @ x1.real), dim=2))
    return y


def parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("model_parameters: {}".format(total))


def Bsplinebao(current_support, ImagSize, basistype):
    Dimmesion = np.shape(current_support)
    B = np.empty(Dimmesion, dtype=object)
    hx = np.zeros(Dimmesion)
    for aa in range(Dimmesion[0]):
        if current_support[aa] == 1:
            B[aa] = 1
            hx[aa] = 0
        else:
            if basistype == 'linear':
                hx[aa] = (np.round(1/2*(current_support[aa] - 1+0.0001)))
            else:
                hx[aa] = (np.round(1/4*(current_support[aa] - 1+0.0001)))
        hx = np.array(hx, dtype=int)
        t1 = np.linspace(0, 1, hx[aa]+1)
        t2 = np.linspace(1, 2, hx[aa] + 1)
        t2 = t2[1:]
        tmp_B = np.zeros(Dimmesion)
        if basistype == 'linear':
            x1 = np.flipud(t1[: -1])
            tmp_B = np.hstack((t1, x1))
        elif basistype == 'cubic':
            x1 = 2 / 3 - np.multiply((1 - np.divide(abs(t1), 2)), np.power(t1, 2))
            x2 = np.power(1 / 6 * (2 - abs(t2)), 3)
            x = np.hstack((x1, x2))
            tmp_B = np.hstack((np.flipud(x[1:]), x))
        B[aa] = tmp_B
    return B, hx


def SDS_BSpline_Basis(B, hx, full_res):
    basis = []
    full_res = np.array(full_res, dtype=int)
    for bb in range(np.size(B)):
        if all(B[bb] == 1):
            basis[bb] = np.ones((1, full_res[bb]))
        else:
            extent = 3 * hx[bb] - 1 + full_res[bb]  # full extent which will be cropped
            num_vecs = math.floor((((extent - 1) - 1) / hx[bb]) + 1)
            last_starting = (num_vecs - 1) * hx[bb] + 1
            support_size = np.size(B[bb])
            tmp_basis = np.zeros((num_vecs, last_starting + support_size - 1))
            for aa in range(num_vecs):
                tmp_basis[aa, aa * hx[bb]: aa * hx[bb] + support_size] = B[bb]
            tmp_basis = tmp_basis[:, 3 * hx[bb]: 3 * hx[bb] + full_res[bb]]
            indx = []
            for aa in range(tmp_basis.shape[0]):
                if np.sum(np.abs(tmp_basis[aa, :])) == 0:
                    indx = np.hstack((indx, aa))
            indx = np.array(indx, dtype=int)
            if np.size(indx) > 0:
               tmp_basis = np.delete(tmp_basis, indx, axis=0)
            basis.append(tmp_basis)
    return basis


def creatBasis(Image):
    b, h, w = Image.size()
    ImagX = h
    ImagY = w
    basisAll = []
    ImagSize = np.array([ImagX, ImagY])
    current_support = np.array([0.0, 0.0])
    full_res = np.array([ImagX, ImagY])
    for iLever in range(8):
        if iLever == 0:
            current_support = np.array([ImagX, ImagY])
        else:
            current_support = np.round(3 * current_support / 4)
        B, hx = Bsplinebao(current_support, ImagSize, 'linear')
        basis = SDS_BSpline_Basis(B, hx, full_res)
        if iLever == 1:
            basisAll.append(basis)
    return basisAll


def creatMap(b, h, w, basis, feature):
    e_PhaseMap = torch.complex(torch.zeros((b, h, w), dtype=torch.float64),
                               torch.zeros((b, h, w), dtype=torch.float64)).cuda()
    PhaseMap = torch.zeros((b, h, w), dtype=torch.float64).cuda()
    for i in range(b):
        for j in range(1):
            output = torch.zeros(h, w).cuda()
            current_support = torch.tensor([h, w])
            for k in range(1):
                if k > 0:
                    current_support = torch.round(3 * current_support / 4)
                tmp = torch.squeeze(feature[k][i][j])
                aa = torch.from_numpy(basis[k][1].T).float().cuda()
                bb = torch.from_numpy(basis[k][0].T).float().cuda()
                tmp = torch.sqrt(1 / current_support[1]) * torch.mm(aa, tmp)
                tmp = torch.sqrt(1 / current_support[0]) * torch.mm(bb, tmp.T)
                output = output + tmp
            PhaseMap[i, :, :] = output
            e_PhaseMap[i, :, :] = torch.cos(torch.div(1j * output, 1j)) - 1j * torch.sin(torch.div(1j * output, 1j))
    del output, current_support, tmp, aa, bb, feature
    return e_PhaseMap, PhaseMap
