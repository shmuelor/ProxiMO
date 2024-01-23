import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import nibabel as nib

from .utils import PhiH_Phi, PhiH, Phi

class UNET(nn.Module):
    def __init__(self, in_channel=6, out_channel=6):
        super(UNET, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv11 = nn.Sequential(nn.Conv3d(self.in_channel, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))

        self.maxpool2m = nn.MaxPool3d(2)
        self.conv21 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.conv22 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))

        self.maxpool3m = nn.MaxPool3d(2)
        self.conv31 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
        self.conv32 = nn.Sequential(nn.Conv3d(128, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))

        self.maxpool4m = nn.MaxPool3d(2)
        self.conv41 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))
        self.conv42 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))

        self.maxpool5m = nn.MaxPool3d(2)
        self.conv51 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=5, padding=2), nn.BatchNorm3d(512), nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv3d(512, 512, kernel_size=5, padding=2), nn.BatchNorm3d(512), nn.ReLU(inplace=True))

        self.deconv61 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, padding=0)
        self.conv62 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))
        self.conv63 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))

        self.deconv71 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0)
        self.conv72 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
        self.conv73 = nn.Sequential(nn.Conv3d(128, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))

        self.deconv81 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
        self.conv82 = nn.Sequential(nn.Conv3d(128, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.conv83 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))

        self.deconv91 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
        self.conv92 = nn.Sequential(nn.Conv3d(64, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.conv93 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))

        self.conv101 = nn.Conv3d(32, self.out_channel, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):

        mul = 2 ** 4
        pad_1, pad_2, pad_3 = 0, 0, 0
        if x.shape[-1] % mul != 0:
            pad_1 = mul - x.shape[-1] % mul
        if x.shape[-2] % mul != 0:
            pad_2 = mul - x.shape[-2] % mul
        if x.shape[-3] % mul != 0:
            pad_3 = mul - x.shape[-3] % mul
        if pad_1 != 0 or pad_2 != 0 or pad_3 != 0:
            x = F.pad(x, (0, pad_1, 0, pad_2, 0, pad_3))
#         print(x.shape)
        
        x1 = self.conv12(self.conv11(x))
        x2 = self.conv22(self.conv21(self.maxpool2m(x1)))
        x3 = self.conv32(self.conv31(self.maxpool3m(x2)))
        x4 = self.conv42(self.conv41(self.maxpool4m(x3)))
        x5 = self.conv52(self.conv51(self.maxpool5m(x4)))

        x = self.conv63(self.conv62(torch.cat((self.deconv61(x5), x4), dim=1)))
        x = self.conv73(self.conv72(torch.cat((self.deconv71(x), x3), dim=1)))
        x = self.conv83(self.conv82(torch.cat((self.deconv81(x), x2), dim=1)))
        x = self.conv93(self.conv92(torch.cat((self.deconv91(x), x1), dim=1)))

        out = self.conv101(x)

        if pad_1 != 0:
            out = out[...,:-pad_1]
        if pad_2 != 0:
            out = out[...,:-pad_2,:]
        if pad_3 != 0:
            out = out[...,:-pad_3,:,:]
        
        return out
        
class LPCNN(nn.Module):
    def __init__(self, gt_mean, gt_std, iter_num, feat_dim, num_blocks):
        super().__init__()
        print('init unet...')
        self.gt_mean = torch.from_numpy(np.load(gt_mean)[:, np.newaxis, np.newaxis, np.newaxis]).float()
        self.gt_std = torch.from_numpy(np.load(gt_std)[:, np.newaxis, np.newaxis, np.newaxis]).float()

        self.iter_num = iter_num
        # feat_dim = 64 # 128
        # num_blocks = 8 # 16

        self.alpha = torch.nn.Parameter(torch.ones(1) * 0.5)
        #self.alpha = 1
        
        self.gen = UNET()
        
        #self.gen = nn.ModuleList([SingleGen() for i in range(self.iter_num)])

    def make_layer(self, block, num_of_layer, **kwargs):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(**kwargs))
        return nn.Sequential(*layers)

    def forward(self, y, dk, mask, ls100, H0=None):
#         self.alpha = torch.nn.Parameter(torch.ones(1).cuda() * 2.5) # set alpha
        
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
        x_est = self.alpha * PhiH(y_padded, dk)[:, :, :x_dim, :y_dim, :z_dim]

        #den_x_pred = ls100
        #print(self.iter_num)
        #print(self.alpha)

        pn_x_pred = torch.zeros_like(x_est)

        for i in range(self.iter_num):
            
            if i == 0:
                pn_x_pred += x_est

            else:
                den_x_pred_padded = F.pad(den_x_pred, (0, pad_z, 0, pad_y, 0, pad_x))
                pn_x_pred = den_x_pred + x_est - self.alpha * PhiH_Phi(den_x_pred_padded, dk, pad_mask)[:, :, :x_dim, :y_dim, :z_dim]
                

#             nib.Nifti1Image(pn_x_pred.cpu().squeeze().permute(1,2,3,0).numpy(), None).to_filename('iter'+str(i)+'_pre.nii.gz')
            
            x_input = ((pn_x_pred - mean) / std) * mask[:, :, 0, :, :, :]
            x_pred = self.gen(x_input)
            den_x_pred = ((x_pred * std) + mean) * mask[:, :, 0, :, :, :]
            
            #out.append(x_pred)

            #den_x_pred = self.gen(pn_x_pred)
            
#             nib.Nifti1Image(den_x_pred.cpu().squeeze().permute(1,2,3,0).numpy(), None).to_filename('iter'+str(i)+'_post.nii.gz')
            
        return x_pred

#   def forward(self, y, dk, mask, ls100):

#       batch_size, _, number, x_dim, y_dim, z_dim = y.shape
#       _, _, _, w_x_dim, w_y_dim, w_z_dim = dk.shape       


#       out = []

#       dk = dk.unsqueeze(-1) # batch, chi(6), orientations, w, h, d, 1
#       dk = dk.index_select(3, torch.cat((torch.arange(0,x_dim//2), torch.arange(w_x_dim-x_dim//2, w_x_dim))).to(dk.device)) \
#                .index_select(4, torch.cat((torch.arange(0,x_dim//2), torch.arange(w_x_dim-x_dim//2, w_x_dim))).to(dk.device)) \
#                .index_select(5, torch.cat((torch.arange(0,x_dim//2), torch.arange(w_x_dim-x_dim//2, w_x_dim))).to(dk.device))

#       mean = self.gt_mean.to(y.device, dtype=torch.float)
#       std = self.gt_std.to(y.device, dtype=torch.float)

#       x_est = self.alpha * PhiH(y, dk)[:, :, :x_dim, :y_dim, :z_dim]

#       #den_x_pred = ls100
#       #print(self.iter_num)
#       #print(self.alpha)

#       pn_x_pred = torch.zeros_like(x_est)

#       for i in range(self.iter_num):
            
#           if i == 0:
#               pn_x_pred += x_est

#           else:
#               pn_x_pred = den_x_pred + x_est - self.alpha * PhiH_Phi(den_x_pred, dk, mask)
                

#           x_input = ((pn_x_pred - mean) / std) * mask[:, :, 0, :, :, :]
#           x_pred = self.gen(x_input)
#           den_x_pred = ((x_pred * std) + mean) * mask[:, :, 0, :, :, :]
            
#           #out.append(x_pred)

#           #den_x_pred = self.gen(pn_x_pred)

#       return x_pred


