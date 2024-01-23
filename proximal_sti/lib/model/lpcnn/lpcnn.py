import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import nibabel as nib

from .utils import PhiH_Phi, PhiH, Phi

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class BasicBlock(nn.Module):

    def __init__(self, inplanes=128, planes=128):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
        
class LPCNN(nn.Module):
    def __init__(self, gt_mean, gt_std, iter_num, feat_dim, num_blocks):
        super().__init__()
        self.gt_mean = torch.from_numpy(np.load(gt_mean)[:, np.newaxis, np.newaxis, np.newaxis]).float()
        self.gt_std = torch.from_numpy(np.load(gt_std)[:, np.newaxis, np.newaxis, np.newaxis]).float()

        self.iter_num = iter_num
        # feat_dim = 64 # 128
        # num_blocks = 8 # 16

        self.alpha = torch.nn.Parameter(torch.ones(1) * 0.5)
        #self.alpha = 1
        
        self.gen = nn.Sequential(
                nn.Conv3d(in_channels=6, out_channels=feat_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                self.make_layer(BasicBlock, num_blocks, inplanes=feat_dim, planes=feat_dim),
                nn.Conv3d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=feat_dim, out_channels=6, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
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


