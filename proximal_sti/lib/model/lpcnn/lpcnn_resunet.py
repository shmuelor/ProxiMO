import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import nibabel as nib

from .unet3d.models import ResidualUNet3D
from lib.StiOperatorToolkit import StiOperatorToolkit as stot
# from utils import pad_for_unet, unpad_for_unet
        
class LPCNN(nn.Module):
    def __init__(self, gt_mean, gt_std, iter_num, feat_dim=None, num_blocks=None, train_step_size=True):
        super().__init__()
        print('init unet...')
        self.gt_mean = torch.from_numpy(np.load(gt_mean)[:, np.newaxis, np.newaxis, np.newaxis]).float()
        self.gt_std = torch.from_numpy(np.load(gt_std)[:, np.newaxis, np.newaxis, np.newaxis]).float()

        self.iter_num = iter_num
        # feat_dim = 64 # 128
        # num_blocks = 8 # 16

        self.alpha = torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=train_step_size)
        #self.alpha = 1
        
        self.gen = ResidualUNet3D(in_channels=6, out_channels=6)

    def make_layer(self, block, num_of_layer, **kwargs):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(**kwargs))
        return nn.Sequential(*layers)

    def forward(self, y, dk, mask, ls100, H0=None):
#         print('alpha=', self.alpha)
        
        batch_size, _, number, x_dim, y_dim, z_dim = y.shape
        _, _, _, w_x_dim, w_y_dim, w_z_dim = dk.shape       
        
        pad_x = w_x_dim - x_dim
        pad_y = w_y_dim - y_dim
        pad_z = w_z_dim - z_dim

        pad_mask = F.pad(mask, (0, pad_z, 0, pad_y, 0, pad_x))


        out = []

        dk = dk.unsqueeze(-1) # batch, chi(6), orientations, w, h, d, 1

        mean = self.gt_mean.to(y.device, dtype=torch.float)
        std = self.gt_std.to(y.device, dtype=torch.float)

        y_padded = F.pad(y, (0, pad_z, 0, pad_y, 0, pad_x))
        x_est = self.alpha * stot.PhiH(y_padded, dk)[:, :, :x_dim, :y_dim, :z_dim]

        #den_x_pred = ls100
        #print(self.iter_num)
        #print(self.alpha)

        pn_x_pred = torch.zeros_like(x_est)

        for i in range(self.iter_num):
            
            if i == 0:
                pn_x_pred += x_est

            else:
                den_x_pred_padded = F.pad(den_x_pred, (0, pad_z, 0, pad_y, 0, pad_x))
                pn_x_pred = den_x_pred + x_est - self.alpha * stot.PhiH_Phi(den_x_pred_padded, dk, pad_mask)[:, :, :x_dim, :y_dim, :z_dim]
                

#             nib.Nifti1Image(pn_x_pred.cpu().squeeze().permute(1,2,3,0).numpy(), None).to_filename('iter'+str(i)+'_pre.nii.gz')
            
            x_input = ((pn_x_pred - mean) / std) * mask[:, :, 0, :, :, :]
            x_input_padded = pad_for_unet(x_input, len(self.gen.encoders))
            x_pred = self.gen(x_input_padded)
            x_pred = unpad_for_unet(x_pred, x_input, len(self.gen.encoders))
            den_x_pred = ((x_pred * std) + mean) * mask[:, :, 0, :, :, :]
            
            #out.append(x_pred)

            #den_x_pred = self.gen(pn_x_pred)
            
#             nib.Nifti1Image(den_x_pred.cpu().squeeze().permute(1,2,3,0).numpy(), None).to_filename('iter'+str(i)+'_post.nii.gz')
            
        return x_pred


def pad_for_unet(x, n_down):
    """
    pad input for unet
    n_down: number of downsampling layers
    """
    mul = 2 ** n_down
    pad_1, pad_2, pad_3 = 0, 0, 0
    if x.shape[-1] % mul != 0:
        pad_1 = mul - x.shape[-1] % mul
    if x.shape[-2] % mul != 0:
        pad_2 = mul - x.shape[-2] % mul
    if x.shape[-3] % mul != 0:
        pad_3 = mul - x.shape[-3] % mul
    if pad_1 != 0 or pad_2 != 0 or pad_3 != 0:
        x = F.pad(x, (0, pad_1, 0, pad_2, 0, pad_3))
    return x

def unpad_for_unet(out, x, n_down):
    """
    unpad output of unet
    n_down: number of downsampling layers
    """
    mul = 2 ** n_down
    pad_1, pad_2, pad_3 = 0, 0, 0
    if x.shape[-1] % mul != 0:
        pad_1 = mul - x.shape[-1] % mul
    if x.shape[-2] % mul != 0:
        pad_2 = mul - x.shape[-2] % mul
    if x.shape[-3] % mul != 0:
        pad_3 = mul - x.shape[-3] % mul
    
    if pad_1 != 0:
        out = out[...,:-pad_1]
    if pad_2 != 0:
        out = out[...,:-pad_2,:]
    if pad_3 != 0:
        out = out[...,:-pad_3,:,:]
    return out

