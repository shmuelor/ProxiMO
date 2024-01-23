import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import nibabel as nib

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

    def forward(self, y, dk, mask, ls100):
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
        
        chem_ex = torch.zeros_like(y[:,:,0:1,:,:,:]) # initialize chemical exchange

        for i in range(self.iter_num):
            
            corrected_y = y - chem_ex
            corrected_y_padded = F.pad(corrected_y, (0, pad_z, 0, pad_y, 0, pad_x))
            corrected_y_padded *= pad_mask
            x_est = self.alpha * PhiH(corrected_y_padded, dk)[:, :, :x_dim, :y_dim, :z_dim]
            if i == 0:
                pn_x_pred += x_est

            else:
                den_x_pred_padded = F.pad(den_x_pred, (0, pad_z, 0, pad_y, 0, pad_x))
                pn_x_pred = den_x_pred + x_est - self.alpha * PhiH_Phi(den_x_pred_padded, dk, pad_mask)[:, :, :x_dim, :y_dim, :z_dim]
                

#             nib.Nifti1Image(pn_x_pred.cpu().squeeze().permute(1,2,3,0).numpy(), None).to_filename('iter'+str(i)+'_pre.nii.gz')
            
            x_input = ((pn_x_pred - mean) / std) * mask[:, :, 0, :, :, :]
            x_pred = self.gen(x_input)
            den_x_pred = ((x_pred * std) + mean) * mask[:, :, 0, :, :, :]
            
            # update chem_ex
            den_x_pred_padded = F.pad(den_x_pred, (0, pad_z, 0, pad_y, 0, pad_x))
            chem_ex = torch.mean(y - Phi(den_x_pred_padded, dk, pad_mask)[:, :, :, :x_dim, :y_dim, :z_dim], 2, keepdim=True)
            
            
            #out.append(x_pred)

            #den_x_pred = self.gen(pn_x_pred)
            
#             nib.Nifti1Image(den_x_pred.cpu().squeeze().permute(1,2,3,0).numpy(), None).to_filename('iter'+str(i)+'_post.nii.gz')
        nib.Nifti1Image(chem_ex.cpu().squeeze().numpy(), None).to_filename('chem_iter'+str(i)+'_post.nii.gz')
        return x_pred


def PhiH_Phi(x, dk, m):
    """
    Implementation of \Phi^H \Phi, where \Phi is forward model, ^H is Hermitian transpose
    Input:
        x: batch, chi(6), w, h, d
        dk: dipole kernel. batch, chi(6), orientations, w, h, d, 1
        m: brain mask. batch, 1, orientations, w, h, d
    """
    x = torch.rfft(x, 3, normalized=True, onesided=False)
    x = x.unsqueeze(2)
    x = dk * x
    x = torch.sum(x, 1, keepdim=True)
    x = torch.ifft(x, 3, normalized=True)[:, :, :, :, :, :, 0]
    x = x * m
    x = torch.rfft(x, 3, normalized=True, onesided=False)
    x = dk * x
    x = torch.sum(x, 2)
    x = torch.ifft(x, 3, normalized=True)[:, :, :, :, :, 0]
    return x


def PhiH(x, dk):
    """
    Implementation of \Phi^H, where \Phi is forward model, ^H is Hermitian transpose.
    Note: Since x is zero outside the brain mask, we omit the mask multiplication step before FFT.
    
    Input:
        x: batch, 1, orientations, w, h, d
        dk: dipole kernel. batch, chi(6), orientations, w, h, d, 1
    """
    x = torch.rfft(x, 3, normalized=True, onesided=False)
    x = dk * x
    x = torch.sum(x, dim=2)
    x = torch.ifft(x, 3, normalized=True)[:, :, :, :, :, 0]
    return x


def Phi(x, dk, m):
    """
    Implementation of \Phi, where \Phi is forward model of STI.
    
    Input:
        x: batch, chi(6), w, h, d
        dk: dipole kernel. batch, chi(6), orientations, w, h, d, 1
        m: mask. batch, 1, orientations, w, h, d
    Return:
        batch, 1, orientations, w, h, d
    """
#     print(x.shape,dk.shape,m.shape)
    x = torch.rfft(x, 3, normalized=True, onesided=False)
    x = x.unsqueeze(2)
    x = dk * x
    x = torch.sum(x, 1, keepdim=True)
    x = torch.ifft(x, 3, normalized=True)[:, :, :, :, :, :, 0]
    x = x * m
    return x



