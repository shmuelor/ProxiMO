import os
import sys

import torch
from torch.utils import data
import torch.nn.functional as F

import numpy as np
import random
import nibabel as nib
import socket

from lib.StiOperatorToolkit import StiOperatorToolkit as stot

class STIDataset(data.Dataset):

    def __init__(self, args, root, device, split='train', sep='partition', tesla=7, number=6, snr=30, is_transform=True, augmentations=None, is_norm=False, patch_size=64, dk_size=0):
        self.root = root
        self.split = split
        self.sep = sep
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.is_norm = is_norm

        self.tesla = tesla
        self.gamma = 42.57747892
        self.number = number
        self.snr = snr

#         self.patch_size = (64, 64, 64)
#         self.whole_size = (224,224,136) # (144, 144, 90) # (128, 128, 128)
        self.patch_size = (patch_size, patch_size, patch_size)
        self.dk_size = (dk_size, dk_size, dk_size)

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
            self.gt_list_file = os.path.join(self.root_path, args.train_list.replace('input', 'gt'))
#           self.input_list_file = os.path.join(self.root_path, split + '_input_one.txt') # single sample train/val
#           self.gt_list_file = os.path.join(self.root_path, split + '_gt_one.txt')
        elif split == 'validate':
            self.input_list_file = os.path.join(self.root_path, args.validate_list)
            self.gt_list_file = os.path.join(self.root_path, args.validate_list.replace('input', 'gt'))
        elif split == 'test':
            self.input_list_file = os.path.join(self.root_path, args.test_list)
            self.gt_list_file = os.path.join(self.root_path, args.test_list.replace('input', 'gt'))
#             self.input_list_file = os.path.join(self.root_path, split + '_input.txt')
#             self.gt_list_file = os.path.join(self.root_path, split + '_gt.txt')
#           self.input_list_file = '/cis/home/zfang23/code/sti_dl/proximal_sti/test_input.txt'
#           self.gt_list_file = '/cis/home/zfang23/code/sti_dl/proximal_sti/test_gt.txt'
        else:
            raise

        self.input_data = []
        self.gt_data = []
        
        with open(self.input_list_file, 'r') as f:
            for line in f:
                self.input_data.append(line.rstrip('\n'))
        
        if self.sep == 'partition':
            self.gt_data = [' '.join(x.split(' ')[:2]) for x in self.input_data]
        elif self.sep == 'whole':
            self.gt_data = [' '.join(x.split(' ')[:1]) for x in self.input_data]  
#         with open(self.gt_list_file, 'r') as f:
#             for line in f:
#                 self.gt_data.append(line.rstrip('\n'))

    def __len__(self):

        return len(self.input_data)

    def comp_convert(self, comp, data):

        if self.sep == 'partition':

            if data == 'phase':
                tensor_numpy = []
                for i in range(self.number):
                    name = ''.join([self.root, self.sep, '/phase_pdata/', comp[0], '/', comp[i+2], '/', comp[0], '_sim_', comp[i+2], '_phase_', 'snr'+str(self.snr)+'_', comp[1], '.npy'])
                    tensor_numpy.append(np.load(name))
                tensor_numpy = np.asarray(tensor_numpy)
                tensor_numpy = tensor_numpy / (self.tesla*self.gamma)

            elif data == 'mask':
                tensor_numpy = []
                for i in range(self.number):
                    name = ''.join([self.root, self.sep, '/mask_pdata/', comp[0], '/', comp[i+2], '/', comp[0], '_sim_', comp[i+2], '_mask_', comp[1], '.npy'])
                    tensor_numpy.append(np.load(name))
                tensor_numpy = np.asarray(tensor_numpy)
                
            elif data == 'dk':
                
                
                name = ''.join([self.root, 'whole/meta_data/', comp[0], '/', comp[0], '_sim', '_sizeVol.npy'])
                sizeVol = np.load(name)
                name = ''.join([self.root, 'whole/meta_data/', comp[0], '/', comp[0], '_sim', '_voxSize.npy'])
                voxSize = np.load(name)
                fov = sizeVol * voxSize
                
                tensor_numpy = []
                for i in range(self.number):

                    name = ''.join([self.root, 'whole/dk_data/', comp[0], '/', comp[i+2], '/', comp[0], '_sim_', comp[i+2], '_dk.npy'])
                    dk = np.load(name).astype('float32')
                    tensor_numpy.append(dk)
                    
#                     if socket.gethostname() == 'ka':
#                         name = ''.join([self.root, 'whole/angle_data/', comp[0], '/', comp[i+2], '/', comp[0], '_sim_', comp[i+2], '_ang.npy'])
#                         H0 = np.load(name)
#                         if self.dk_size[0] == 0:
#                             dk_size = sizeVol
#                         else:
#                             dk_size = self.dk_size
#                         dk2 = angle2dk(H0, dk_size, fov)
#                         dk2 = np.swapaxes(dk2, 0, 1)
#                         tensor_numpy.append(dk2)
                    
#                     print('err', np.mean(np.abs(dk-dk2)[:]))
                    
#                     print('loaded dk.')
                tensor_numpy = np.asarray(tensor_numpy)
#                 print('all loaded dk.')

            elif data == 'gt':
                name = ''.join([self.root, self.sep, '/sti_pdata/', comp[0], '/', comp[0], '_sim_tensor_', comp[1], '.npy'])
                tensor_numpy = np.load(name)

            elif data == 'ani':
                name = ''.join([self.root, self.sep, '/ani_pdata/', comp[0], '/', comp[0], '_sim_ani_', comp[1], '.npy'])
                tensor_numpy = np.load(name)

            elif data == 'ls100':
                ori_list = []
                for i in range(self.number):
                    ori_list.append(comp[i+2])
                ori = ''.join(ori_list)
                name = ''.join([self.root, self.sep, '/ls100_pdata/', comp[0], '/', ori, '/', comp[0], '_sim_', ori, '_ls100_', comp[1], '.npy'])
                tensor_numpy = np.load(name) 
    
            elif data == 'name':
                ori_list = []
                for i in range(self.number):
                    ori_list.append(comp[i+2])
                ori = ''.join(ori_list)
                tensor_numpy = ''.join([comp[0], '_', ori, '_', comp[1]])

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
        
            elif data == 'ls100':
                ori_list = []
                for i in range(self.number):
                    ori_list.append(comp[i+1])
                ori = ''.join(ori_list)
                name = ''.join([self.root, self.sep, '/ls100_data/', comp[0], '/', ori, '/', comp[0], '_sim_', ori, '_ls100.npy'])
                tensor_numpy = np.load(name)

            elif data == 'name':
                ori_list = []
                for i in range(self.number):
                    ori_list.append(comp[i+1])
                ori = '-'.join(ori_list)
                tensor_numpy = ''.join([comp[0], '_', ori])

        return tensor_numpy

    def __getitem__(self, index):

        input_comp_list = self.input_data[index].split(' ')
        gt_comp = self.gt_data[index].split(' ')

        input_tensor_list = self.comp_convert(input_comp_list, 'phase')
        mask_tensor_list = self.comp_convert(input_comp_list, 'mask')
        dk_tensor_list = self.comp_convert(input_comp_list, 'dk')
        gt_tensor = self.comp_convert(gt_comp, 'gt')
        ani_tensor = self.comp_convert(gt_comp, 'ani')
        #ls100_tensor = self.comp_convert(input_comp_list, 'ls100')

        if self.sep == 'partition' and socket.gethostname() == 'ka':
            x = torch.randint(0,input_tensor_list.shape[-3]-self.patch_size[0]+1,(1,))
            y = torch.randint(0,input_tensor_list.shape[-2]-self.patch_size[1]+1,(1,))
            z = torch.randint(0,input_tensor_list.shape[-1]-self.patch_size[2]+1,(1,))
    #         print(x,y,z)
            input_tensor_list = input_tensor_list[...,x:x+self.patch_size[0],y:y+self.patch_size[1],z:z+self.patch_size[2]]
            mask_tensor_list = mask_tensor_list[...,x:x+self.patch_size[0],y:y+self.patch_size[1],z:z+self.patch_size[2]]
            gt_tensor = gt_tensor[...,x:x+self.patch_size[0],y:y+self.patch_size[1],z:z+self.patch_size[2],:]
            ani_tensor = ani_tensor[...,x:x+self.patch_size[0],y:y+self.patch_size[1],z:z+self.patch_size[2]]
    #         print(input_tensor_list.shape, mask_tensor_list.shape, gt_tensor.shape, ani_tensor.shape, dk_tensor_list.shape)
        
        
        sub_name = self.comp_convert(input_comp_list, 'name')
        print(sub_name)

        if self.is_norm:
            gt_tensor = ((gt_tensor - self.gt_mean) / self.gt_std) * mask_tensor_list[0, :, :, :][:, :, :, np.newaxis]
            #ls100_tensor = ((ls100_tensor - self.gt_mean) / self.gt_std) * mask_tensor_list[0, :, :, :][:, :, :, np.newaxis]

        input_tensor_list = input_tensor_list[np.newaxis, :, :, :, :].astype('float32')
        gt_tensor = np.moveaxis(gt_tensor, -1, 0).astype('float32')
        #ls100_tensor = np.moveaxis(ls100_tensor, -1, 0)
        dk_tensor_list = np.moveaxis(dk_tensor_list, -1, 0).astype('float32')

        mask_tensor_list = mask_tensor_list.astype('float32')
        ani_tensor = ani_tensor.astype('float32')

#       self.sanity_check(input_tensor_list, mask_tensor_list, dk_tensor_list, gt_tensor, ani_tensor)

        return input_tensor_list, gt_tensor, mask_tensor_list, dk_tensor_list, ani_tensor, sub_name, sub_name

    def sanity_check(self, input_tensor, mask, dk, gt, ani):
        gt = (np.moveaxis((np.moveaxis(gt, 0, -1) * self.gt_std + self.gt_mean), -1, 0) * mask).astype('float32') #denormalize
        input_tensor = torch.from_numpy(input_tensor)
        mask = torch.from_numpy(mask)
        dk = torch.from_numpy(dk)
        gt = torch.from_numpy(gt)
        ani = torch.from_numpy(ani)

        print(input_tensor.shape, mask.shape, dk.shape, gt.shape, ani.shape)
        whole_size = dk.shape[-1]
        patch_size = mask.shape[-1]
        pad_gt = F.pad(gt, (0, whole_size-patch_size, 0, whole_size-patch_size, 0, whole_size-patch_size))
        pad_mask = F.pad(mask, (0, whole_size-patch_size, 0, whole_size-patch_size, 0, whole_size-patch_size))
        measure = stot.Phi(pad_gt.unsqueeze(0), dk.unsqueeze(0).unsqueeze(-1), pad_mask.unsqueeze(0).unsqueeze(0))
        measure = measure[0,0,:,:patch_size,:patch_size,:patch_size]
        input_tensor = input_tensor[0,:,:,:,:]
        print(((measure - input_tensor) ** 2).mean()**0.5)
        print(input_tensor.mean(), input_tensor.std(), input_tensor.max(), input_tensor.min())
        print(corr(measure, measure), corr(measure, input_tensor))
        print('psnr', psnr_sti(np.moveaxis(measure.numpy(), 0, -1), np.moveaxis(input_tensor.numpy(), 0, -1), mask[0,:,:,:]))
        nib.Nifti1Image(measure[0,:,:,:].numpy(), None).to_filename('m.nii.gz')
        nib.Nifti1Image(input_tensor[0,:,:,:].numpy(), None).to_filename('ref.nii.gz')

def corr(a, b):
    a = a.flatten()
    b = b.flatten()
    return ((a - a.mean())*(b-b.mean())).mean()/(a.std()*b.std())

def psnr_sti(a, b, mask):
    """
    psnr with b as ground-truth
    Input:
        a, b: (w,h,d,6)
        mask: (w,h,d)
    Output:
        scalar
    """
    assert a.shape[-1] == 6 and b.shape[-1] == 6
    data_range = b.max() - b.min() # max{all data} - min{all data}
    max_sig_power = (data_range)**2
#     noise_power = np.mean((a-b)**2)
    noise_power = np.average(np.mean((a-b)**2,axis=-1), weights=mask)
    psnr = 10*np.log10(max_sig_power/noise_power)
    return psnr
    