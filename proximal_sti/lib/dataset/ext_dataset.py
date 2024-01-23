import os
import sys

import torch
from torch.utils import data
import torch.nn.functional as F

import numpy as np
import random
import nibabel as nib
import scipy.io
import h5py

from lib.StiOperatorToolkit import StiOperatorToolkit as stot

class ExtDataset(data.Dataset):

    def __init__(self, data_info, device, is_norm=False):
        """
        data_info: list of dict. each entry contains:
            name: str
            tesla: float
            phase: [str]
            dk: [str]
            mask: str
            gt: str
            ani: str
        """
        self.data_info = data_info
        self.device = device
        self.is_norm = is_norm

        self.gamma = 42.57747892
        

    def __len__(self):
        return len(self.data_info)

    def set_mean_std(self, gt_mean, gt_std):
        self.gt_mean = gt_mean
        self.gt_std = gt_std

        self.gt_mean_torch = torch.from_numpy(self.gt_mean).to(self.device, dtype=torch.float).view(1, -1, 1, 1, 1)
        self.gt_std_torch = torch.from_numpy(self.gt_std).to(self.device, dtype=torch.float).view(1, -1, 1, 1, 1)
        

    def __getitem__(self, index):
        
        data = self.data_info[index]
        tesla = data['tesla']
        
        tensor_numpy = [myload(x) for x in data['phase']]
        tensor_numpy = np.array(tensor_numpy)
        tensor_numpy = tensor_numpy / (tesla*self.gamma)
        input_tensor_list = tensor_numpy
        
        # load dipole kernel
        if 'dk' in data:
            tensor_numpy = [myload(x) for x in data['dk']]
            tensor_numpy = np.array(tensor_numpy)
            dk_tensor_list = tensor_numpy
        elif 'H0' in data:
            """Generate dipole kernel from H0"""
            # load voxel size from nii file
            voxSize = nib.load(data['phase'][0]).header.get_zooms()
            tensor_numpy = []
            for x in data['H0']:
                H0 = myload(x)
                sizeVol = np.array(input_tensor_list.shape[-3:])
                fov = sizeVol * voxSize
                print(f"sizeVol: {sizeVol}, voxSize: {voxSize}, fov: {fov}, H0:{H0}")
                tensor_numpy.append(stot.angle2dk_LPS(H0, sizeVol, fov))
            tensor_numpy = np.array(tensor_numpy)
            dk_tensor_list = tensor_numpy
        else:
            raise('Either dk or H0 is needed.')
        
        if isinstance(data['mask'], list):
            tensor_numpy = [myload(x) for x in data['mask']]
        else:
            tensor_numpy = [myload(x) for x in [data['mask']]*len(data['phase'])]
        tensor_numpy = np.array(tensor_numpy)
        mask_tensor_list = tensor_numpy
        
        if 'gt' in data:
            gt_tensor = myload(data['gt'])
        else:
            gt_tensor = np.empty(input_tensor_list.shape[1:]+(6,)) * np.NaN
        if 'ani' in data:
            ani_tensor = myload(data['ani'])
        else:
            ani_tensor = np.empty(input_tensor_list.shape[1:]) * np.NaN
        
        sub_name = data['name']
        
        if self.is_norm:
            gt_tensor = ((gt_tensor - self.gt_mean) / self.gt_std) * mask_tensor_list[0, :, :, :][:, :, :, np.newaxis]
        
        input_tensor_list = input_tensor_list[np.newaxis, :, :, :, :].astype('float32')
        gt_tensor = np.moveaxis(gt_tensor, -1, 0).astype('float32')
        dk_tensor_list = np.moveaxis(dk_tensor_list, -1, 0).astype('float32')
        mask_tensor_list = mask_tensor_list.astype('float32')
        ani_tensor = ani_tensor.astype('float32')
        
        return input_tensor_list, gt_tensor, mask_tensor_list, dk_tensor_list, ani_tensor, index, sub_name

def myload(x):
    if x.endswith('.npy'):
        return np.load(x)
    elif x.endswith('.nii.gz'):
        y = nib.load(x).get_fdata().astype('float32')
#         print('img', y.shape)
        return y
    elif x.endswith('.mat'):
        # dipole tensor
        y = scipy.io.loadmat(x)['dipole_tensor']
        y = np.swapaxes(y, 0, 1)
#         print(y.shape)
        return y.astype('float32')
    elif x.endswith('.h5'):
        # dipole tensor
        with h5py.File(x, 'r') as f:
            y = f['dipole_tensor'][:]
            y = np.transpose(y, [3,2,1,0])
            y = np.swapaxes(y, 0, 1)
#             print(y.shape, y.dtype)
        return y.astype('float32')
    elif x.endswith('.txt'):
        # H0
        with open(x, 'r') as f:
            y = [_.rstrip() for _ in f.readlines()]
            y = np.array(y).astype('float32')
        return y
    