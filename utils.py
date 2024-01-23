import numpy as np
import torch

from drop_pix import DropPixelOperator
from proximal_sti.lib.QsmOperatorToolkit import QsmOperator

def sample_A(type='DropPixel', scaling=3., size_vol=[64]*3):
    if type == 'DropPixel':
        A = DropPixelOperator(p=0.2, size=[64, 64, 64])
        return A
    elif type == 'Qsm':
        H0 = np.random.randn(3)
        H0 = H0 / np.linalg.norm(H0)
        A = QsmOperator.from_H0(
            H0=np.stack([H0], axis=0), 
            meta={'sizeVol': size_vol, 'voxSize': [0.982, 0.982, 1.]}, 
            mask=np.ones(size_vol),
            scaling=scaling)
        return A

def construct_A(type='DropPixel', Phi=None, scaling=3., size_vol=[64]*3):
    if type == 'DropPixel':
        raise NotImplementedError
    elif type == 'Qsm':
        H0 = Phi['H0']
        A = QsmOperator.from_H0(
            H0=np.stack([H0], axis=0), 
            meta={'sizeVol': size_vol, 'voxSize': [0.982, 0.982, 1.]}, 
            mask=np.ones(size_vol),
            scaling=scaling)
        return A

def sample_snr(lo, hi):
    # torch random sample
    return torch.rand(1).item() * (hi - lo) + lo

def get_dk_shape(sub):
    if sub in [1, 2, 3, 4]:
        return [224, 224, 126]
    elif sub in [5, 6, 8, 9]:
        return [224, 224, 110]
    elif sub in [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]:
        return [224, 256, 176]
    elif sub > 1000:
        return [224, 224, 140]
    else:
        print(sub)
        raise ValueError
    
def denormalize(x, mean, std, mask):
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std)
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean)
    x = x * std.to(x.device)
    x = x + mean.to(x.device)
    x = x * mask
    return x

def normalize(x, mean, std, mask):
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std)
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean)
    x = x - mean.to(x.device)
    x = x / std.to(x.device)
    x = x * mask
    return x