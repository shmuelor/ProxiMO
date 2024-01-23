import os
import sys

import torch
from torch.utils import data
import torch.nn.functional as F

import numpy as np
import random
import nibabel as nib
import socket
import torchio as tio

from lib.StiOperatorToolkit import StiOperatorToolkit as stot
from lib.dataset.utils import IsotropicLesionAugmentation as ila
from lib.dataset.augmentation import get_transform


class STIDatasetEfficient(data.Dataset):

    def __init__(self, args, root, device, split='train', sep='partition', tesla=7, number=6, snr=30, is_transform=True, augmentations=None, is_norm=False, patch_size=64, dk_size=0):
        self.root = root
        self.split = split
        self.sep = sep
        self.is_norm = is_norm

        self.tesla = tesla
        self.gamma = 42.57747892
        self.number = number
        self.snr = snr
        self.is_aug = args.is_aug
        self.random_nori = args.random_nori
        self.training_transform = get_transform()

#         self.patch_size = (64, 64, 64)
#         self.whole_size = (224,224,136) # (144, 144, 90) # (128, 128, 128)

        if self.sep == 'partition':
            self.root_path = self.root + self.sep + '/partition_data{}_list/'.format(self.number)
        elif self.sep == 'whole':
            self.root_path = self.root + self.sep + '/data{}_list/'.format(self.number)

        self.gt_mean = None
        self.gt_std = None

        self.gt_mean_torch = None
        self.gt_std_torch = None

        if self.is_norm:

            gt_mean_name = 'train_gt_mean.npy'
            gt_std_name = 'train_gt_std.npy'

            self.gt_mean = np.load(os.path.join(self.root, '..', 'meta', 'stats', gt_mean_name))
            self.gt_std = np.load(os.path.join(self.root, '..', 'meta', 'stats', gt_std_name))

            self.gt_mean_torch = torch.from_numpy(self.gt_mean).to(device, dtype=torch.float).view(1, -1, 1, 1, 1)
            self.gt_std_torch = torch.from_numpy(self.gt_std).to(device, dtype=torch.float).view(1, -1, 1, 1, 1)

        #get data path
        if split == 'train':
            self.input_list_file = os.path.join(self.root_path, args.train_list)
        elif split == 'validate':
            self.input_list_file = os.path.join(self.root_path, args.validate_list)
        elif split == 'test':
            self.input_list_file = os.path.join(self.root_path, args.test_list)
        else:
            raise

        self.input_data = []
        with open(self.input_list_file, 'r') as f:
            for line in f:
                self.input_data.append(line.rstrip('\n'))
        
        if self.sep == 'partition':
            self.gt_data = [' '.join(x.split(' ')[:2]) for x in self.input_data]
        elif self.sep == 'whole':
            self.gt_data = [' '.join(x.split(' ')[:1]) for x in self.input_data]  


    def __len__(self):

        return len(self.input_data)

    def comp_convert(self, comp, data):

        if self.sep == 'partition':

            if data == 'mask':
                tensor_numpy = []
                for i in range(self.number):
                    name = ''.join([self.root, self.sep, '/mask_pdata/', comp[0], '/', comp[i+2], '/', comp[0], '_sim_', comp[i+2], '_mask_', comp[1], '.npy'])
                    tensor_numpy.append(np.load(name))
                tensor_numpy = np.asarray(tensor_numpy)
            
            elif data == 'gt':
                name = ''.join([self.root, self.sep, '/sti_pdata/', comp[0], '/', comp[0], '_sim_tensor_', comp[1], '.npy'])
                tensor_numpy = np.load(name)

            elif data == 'ani':
                name = ''.join([self.root, self.sep, '/ani_pdata/', comp[0], '/', comp[0], '_sim_ani_', comp[1], '.npy'])
                tensor_numpy = np.load(name)

            elif data == 'H0':
                tensor_numpy = []
                for i in range(self.number):
                    name = ''.join([self.root, 'whole/angle_data/', comp[0], '/', comp[i+2], '/', comp[0], '_sim_', comp[i+2], '_ang.npy'])
                    H0 = np.load(name)
                    tensor_numpy.append(H0)
    
            elif data == 'name':
                ori_list = []
                for i in range(self.number):
                    ori_list.append(comp[i+2])
                ori = ''.join(ori_list)
                tensor_numpy = ''.join([comp[0], '_', ori, '_', comp[1]])
            
            elif data == 'meta':
                meta = {}
                name = ''.join([self.root, 'whole/meta_data/', comp[0], '/', comp[0], '_sim_sizeVol.npy'])
                meta['sizeVol'] = np.load(name)
                name = ''.join([self.root, 'whole/meta_data/', comp[0], '/', comp[0], '_sim_voxSize.npy'])
                meta['voxSize'] = np.load(name)
                tensor_numpy = meta

        elif self.sep == 'whole':

            if data == 'phase':
#                 tensor_numpy = np.empty((self.number,) + self.whole_size)
                tensor_numpy = []
                for i in range(self.number):
                    name = ''.join([self.root, self.sep, '/phase_data/', comp[0], '/', comp[i+1], '/', comp[0], '_sim_', comp[i+1], '_phase_', 'snr'+str(self.snr), '.npy'])
                    #name = ''.join([self.root, self.sep, '/phase_data/', comp[0], '/', comp[i+1], '/', comp[0], '_sim_', comp[i+1], '_phase.npy'])
                    tensor_numpy.append(np.load(name))
                tensor_numpy = np.array(tensor_numpy)
                tensor_numpy = tensor_numpy / (self.tesla*self.gamma)

            elif data == 'mask':
#                 tensor_numpy = np.empty((self.number,) + self.whole_size)
                tensor_numpy = []
                for i in range(self.number):
                    name = ''.join([self.root, self.sep, '/mask_data/', comp[0], '/', comp[0], '_sim_mask.npy'])
                    tensor_numpy.append(np.load(name))
                tensor_numpy = np.array(tensor_numpy)

            elif data == 'dk':
#                 tensor_numpy = np.empty((self.number,) + self.whole_size + (6,))
                tensor_numpy = []
                
                for i in range(self.number):
                    name = ''.join([self.root, 'whole/dk_data/', comp[0], '/', comp[i+1], '/', comp[0], '_sim_', comp[i+1], '_dk.npy'])
                    tensor_numpy.append(np.load(name))
                tensor_numpy = np.array(tensor_numpy)
                

            elif data == 'gt':
                name = ''.join([self.root, self.sep, '/sti_data/', comp[0], '/', comp[0], '_sim_tensor.npy'])
                tensor_numpy = np.load(name)

            elif data == 'ani':
                name = ''.join([self.root, self.sep, '/ani_data/', comp[0], '/', comp[0], '_sim_ani.npy'])
                tensor_numpy = np.load(name)
        
            elif data == 'H0':
                tensor_numpy = []
                for i in range(self.number):
                    name = ''.join([self.root, 'whole/angle_data/', comp[0], '/', comp[i+1], '/', comp[0], '_sim_', comp[i+1], '_ang.npy'])
                    H0 = np.load(name)
                    tensor_numpy.append(H0)

            elif data == 'name':
                ori_list = []
                for i in range(self.number):
                    ori_list.append(comp[i+1])
                ori = '-'.join(ori_list)
                tensor_numpy = ''.join([comp[0], '_', ori])

        return tensor_numpy

    def aug_gt(self, sample):
        
        if np.random.rand() < 0.5:
#             ncub = np.random.randint(1,4)
            ncub = 3
            sample['gt'], ma = ila.add_isotropic_cuboid(sample['gt'], sample['brain_mask'], 
                                                        ncub=ncub, cubszlb=6, cubszub=18, returnM=True)
            
        
        sample['gt'] = np.transpose(sample['gt'], [3,0,1,2]) # input dimensions must be [C,W,H,D]
        sample['brain_mask'] = sample['brain_mask'][None,:,:,:]
        sample_tio = tio.Subject(gt=tio.ScalarImage(tensor=sample['gt']), brain_mask=tio.LabelMap(tensor=sample['brain_mask']))
        sample_tio = self.training_transform(sample_tio)
        sample = {'gt': sample_tio['gt'].numpy(), 'brain_mask': sample_tio['brain_mask'].numpy()}
        sample['gt'] = np.array(np.transpose(sample['gt'], [1,2,3,0]))
        sample['brain_mask'] = np.array(sample['brain_mask'][0,:,:,:])
        
        return sample
    
    @staticmethod
    def gen_dk_random(H0_list, dk_sz, voxSize):
        """
        H0 in [RL, AP, IS] orientation!!!
        dk_sz, voxSize all in [RL, AP, IS] orientation!!!
        Return: in [RL, AP, IS] orientation.
        """
        fov = [dk_sz[0] * voxSize[0], dk_sz[1] * voxSize[1], dk_sz[2] * voxSize[2]]
        
        dk = np.zeros([len(H0_list), dk_sz[0], dk_sz[1], dk_sz[2], 6], dtype='float32')
        
        nori = np.random.choice([1,2,3,4,5,6])
        H0_list_new = [H0_list[i] for i in np.random.choice(len(H0_list), nori, replace=False)]
        
        for i in range(len(H0_list_new)):
            dk[i, :, :, :, :] = stot.angle2dk_LPS(H0_list_new[i], dk_sz, fov)
        return dk
        
        
    
    @staticmethod
    def gen_dk(H0_list, sizeVol, voxSize):
        """
        H0 in [RL, AP, IS] orientation!!!
        sizeVol, voxSize all in [RL, AP, IS] orientation!!!
        Return: in [RL, AP, IS] orientation.
        """
        fov = [sizeVol[0] * voxSize[0], sizeVol[1] * voxSize[1], sizeVol[2] * voxSize[2]]
        
        dk = np.zeros([len(H0_list), sizeVol[0], sizeVol[1], sizeVol[2], 6], dtype='float32')
        for i in range(len(H0_list)):
            dk[i, :, :, :, :] = stot.angle2dk_LPS(H0_list[i], sizeVol, fov)
        return dk
    
    @staticmethod
    def fwd_sti(sample, snr):
        dk_size = sample['dk'].shape[1:4]
        
        x = torch.from_numpy(sample['gt']).permute(3, 0, 1, 2).unsqueeze(0)
        dk = torch.from_numpy(sample['dk']).permute(4, 0, 1, 2, 3).unsqueeze(0).unsqueeze(-1)
        m = torch.from_numpy(sample['brain_mask']).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        osz = x.shape[-3:] # original size
        pad = (0, dk_size[-1]-x.shape[-1], 0, dk_size[-2]-x.shape[-2], 0, dk_size[-3]-x.shape[-3])
        x = F.pad(x, pad)
        m = F.pad(m, pad)
        
        meas = stot.Phi(x, dk, m).squeeze(0).squeeze(0).numpy()
        meas = meas[:,:osz[0],:osz[1],:osz[2]]
        
        noise_pw = np.mean(meas[:,sample['brain_mask']==1]**2) / 10 ** (snr / 10) # noise power
        meas = meas + np.random.normal(size=meas.shape) * np.sqrt(noise_pw)
        meas = meas * sample['brain_mask'][None,:,:,:]
        
        return meas
        
    def __getitem__(self, index):
        if self.split == 'train':
            input_comp_list = self.input_data[index].split(' ')
            gt_comp = self.gt_data[index].split(' ')

            gt_tensor = self.comp_convert(gt_comp, 'gt')
            mask_tensor_list = self.comp_convert(input_comp_list, 'mask')
            ani_tensor = self.comp_convert(gt_comp, 'ani')
            H0_list = self.comp_convert(input_comp_list, 'H0')
            meta = self.comp_convert(input_comp_list, 'meta')

            sample = {}
            sample['gt'] = gt_tensor
            sample['brain_mask'] = mask_tensor_list[0,:,:,:]

            # Augment gt, e.g., add isotropic lesions
            if self.is_aug:
                sample = self.aug_gt(sample)
                

            # Generate dipole kernel from H0's
            # dk_sz = meta['sizeVol']
            dk_sz = sample['gt'].shape[0:3]
            if self.random_nori:
                dk_tensor_list = self.gen_dk_random(H0_list, dk_sz, meta['voxSize']) # shape: [ori, x, y, z, 6]
            else:
                dk_tensor_list = self.gen_dk(H0_list, dk_sz, meta['voxSize']) # shape: [ori, x, y, z, 6]
            sample['dk'] = dk_tensor_list

            # Generate measurements from gt and dipole kernel
            input_tensor_list = self.fwd_sti(sample, self.snr)
            sample['meas'] = input_tensor_list # [ori, x, y, z]

            sub_name = self.comp_convert(input_comp_list, 'name')
            print(sub_name)

            if self.is_norm:
                sample['gt'] = ((sample['gt'] - self.gt_mean) / self.gt_std) * sample['brain_mask'][:, :, :, np.newaxis]

            sample['meas'] = sample['meas'][np.newaxis, :, :, :, :].astype('float32')
            sample['gt'] = np.moveaxis(sample['gt'], -1, 0).astype('float32') # to [6, x, y, z]
            sample['dk'] = np.moveaxis(sample['dk'], -1, 0).astype('float32') # to [6, ori, x, y, z]
            sample['brain_mask'] = sample['brain_mask'].astype('float32')
            sample['ani'] = ani_tensor.astype('float32')

    #       self.sanity_check(input_tensor_list, mask_tensor_list, dk_tensor_list, gt_tensor, ani_tensor)

            return sample['meas'], sample['gt'], np.repeat(sample['brain_mask'][None,:,:,:], len(H0_list), axis=0), sample['dk'], sample['ani'], sub_name, sub_name
        
        
        elif self.split in ['validate', 'test']:
            """ No augmentation (no changing of ground-truth and measurements). """
            print('original loading')
            input_comp_list = self.input_data[index].split(' ')
            gt_comp = self.gt_data[index].split(' ')

            input_tensor_list = self.comp_convert(input_comp_list, 'phase')
            mask_tensor_list = self.comp_convert(input_comp_list, 'mask')
            dk_tensor_list = self.comp_convert(input_comp_list, 'dk')
            gt_tensor = self.comp_convert(gt_comp, 'gt')
            ani_tensor = self.comp_convert(gt_comp, 'ani')

            sub_name = self.comp_convert(input_comp_list, 'name')
            print(sub_name)

            if self.is_norm:
                gt_tensor = ((gt_tensor - self.gt_mean) / self.gt_std) * mask_tensor_list[0, :, :, :][:, :, :, np.newaxis]

            input_tensor_list = input_tensor_list[np.newaxis, :, :, :, :].astype('float32')
            gt_tensor = np.moveaxis(gt_tensor, -1, 0).astype('float32')
            dk_tensor_list = np.moveaxis(dk_tensor_list, -1, 0).astype('float32')
            mask_tensor_list = mask_tensor_list.astype('float32')
            ani_tensor = ani_tensor.astype('float32')

    #       self.sanity_check(input_tensor_list, mask_tensor_list, dk_tensor_list, gt_tensor, ani_tensor)

            return input_tensor_list, gt_tensor, mask_tensor_list, dk_tensor_list, ani_tensor, sub_name, sub_name


        else:
            raise

    