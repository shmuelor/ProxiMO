import numpy as np
import torch
from torch import nn

from .StiOperatorToolkit import _rfft, _ifft


def angle2dk(H0, sizeVol, fov):
    """
    From the direction of magnetic field in subject frame of reference (H0)
    to dipole kernel.
    Use LPS orientation for all.

    Inputs:
        H0: Unit vector, direction of magnetic field in [RL, AP, IS] (LPS orientation). 
            RL: subject right-left. AP: Anterior-Posterior. IS: Inferior-Superior.
        sizeVol: size of dipole kernel. [RL, AP, IS] (LPS orientation).
        fov: field of view in mm. [RL, AP, IS] (LPS orientation).
    Return:
        dk: dipole kernel. LPS.
                       [RL, AP, IS].
    """
    Nx = sizeVol[0]
    Ny = sizeVol[1]
    Nz = sizeVol[2]

    dkx = 1/fov[0]
    dky = 1/fov[1]
    dkz = 1/fov[2]

    # convolution kernel 

    def Ni2linspace(Ni):
        if Ni % 2 == 0:
            pts = np.linspace(-Ni/2, Ni/2-1, Ni);
        else:
            pts = np.linspace(-(Ni-1)/2, (Ni-1)/2, Ni);
        return pts

    kx = Ni2linspace(Nx) * dkx;
    ky = Ni2linspace(Ny) * dky;
    kz = Ni2linspace(Nz) * dkz;

    kx = kx.astype('float64')
    ky = ky.astype('float64')
    kz = kz.astype('float64')

    KX_Grid, KY_Grid, KZ_Grid = np.meshgrid(kx, ky, kz, indexing='ij') # mesh in k space
    KSq = KX_Grid ** 2 + KY_Grid ** 2 + KZ_Grid ** 2          # k^2

    hx, hy, hz = H0

    B0 = (hx*KX_Grid + hy*KY_Grid + hz*KZ_Grid)/KSq
    dk = (1/3)*(hx*hx) - B0*KX_Grid*hx + \
         (1/3)*(hy*hy) - B0*KY_Grid*hy + \
         (1/3)*(hz*hz) - B0*KZ_Grid*hz

    dk = np.fft.ifftshift(dk)

    dk[np.isnan(dk)] = 0

    dk = dk.astype('float32')

    return dk

def Phi(x, dk, m):
    """
    Implementation of \Phi, where \Phi is forward operator of QSM.
    y = \Phi * x = mask * F^-1 * D * F * x.
    All inputs and outputs are torch.Tensor.

    Input:
        x: batch, w, h, d. 
        dk: dipole kernel. batch, orientations, w, h, d
        m: mask. batch, orientations (or 1), w, h, d
    Return:
        batch, orientations, w, h, d
    """
#     print(x.shape,dk.shape,m.shape)
    x = _rfft(x)
    x = x.unsqueeze(1)
    x = dk.unsqueeze(-1) * x
    x = _ifft(x)[:, :, :, :, :, 0]
    x = x * m
    return x

def PhiH(x, dk, m):
    """
    Implementation of \Phi^H, where \Phi is forward operator of QSM, ^H is Hermitian transpose.
    y = \Phi^H * x = F^-1 * D^H * F * mask^H * x.ß

    Input:
        x: batch, orientations, w, h, d
        dk: dipole kernel. batch, orientations, w, h, d
        m: mask. batch, orientations (or 1), w, h, d
    Return:
        batch, w, h, d. 
    """
    x = x * m
    x = _rfft(x)
    x = dk.unsqueeze(-1) * x
    x = torch.sum(x, dim=1)
    x = _ifft(x)[:, :, :, :, 0]
    return x

def PhiH_Phi(x, dk, m):
    """
    Implementation of \Phi^H \Phi, where \Phi is forward operator of QSM, ^H is Hermitian transpose.

    Input:
        x: batch, w, h, d
        dk: dipole kernel. batch, orientations, w, h, d
        m: brain mask. batch, orientations (or 1), w, h, d
    Return:
        batch, w, h, d. 
    """
    x = _rfft(x)
    x = x.unsqueeze(1)
    x = dk.unsqueeze(-1) * x
    # x = _ifft(x)[:, :, :, :, :, 0]
    # x = x * m
    # x = _rfft(x)
    x = dk.unsqueeze(-1) * x
    x = torch.sum(x, dim=1)
    x = _ifft(x)[:, :, :, :, 0]
    return x


def PhiH_Phi_plus_epsilonI_inv_PhiH(y, dk, m, epsilon=1e-6):
    """
    Implementation of (\Phi^H \Phi + \eps I)^{-1} \Phi^H, where \Phi is forward operator of QSM, ^H is Hermitian transpose.
    epsilon is a small number to avoid singularity and can be viewed as l2 regularization.

    Input:
        y: batch, orientations, w, h, d
        dk: dipole kernel. batch, orientations, w, h, d
        m: brain mask. batch, orientations (or 1), w, h, d
    Return:
        batch, w, h, d. 
    """
    x = y * m
    x = _rfft(x)
    x = dk.unsqueeze(-1) * x
    x = torch.sum(x, dim=1) # batch, w, h, d, 2

    x = x / (torch.sum(dk ** 2, dim=1) + epsilon).unsqueeze(-1) # batch, w, h, d, 2
    x = _ifft(x)[:, :, :, :, 0]

    return x


class QsmOperator(object):
    def __init__(self, H0=None, meta=None, dk=None, mask=None, scaling=None):
        
        self.dk = dk # nori, nx, ny, nz
        self.mask = mask # nx, ny, nz
        self.H0 = H0 # nori, 3
        self.meta = meta # dict
        self.scaling = scaling # float

    @classmethod
    def from_H0(cls, H0, meta, mask, scaling=1.):
        """
        Inputs:
            H0: (nori, 3). Unit vectors, direction of magnetic field in [RL, AP, IS] (LPS orientation). 
                RL: subject right-left. AP: Anterior-Posterior. IS: Inferior-Superior.
            meta: meta data of the image. dict.
                'sizeVol': size of dipole kernel. [RL, AP, IS] (LPS orientation).
                'voxSize': voxel size in mm. [RL, AP, IS] (LPS orientation).
            mask: (nx, ny, nz). brain mask. LPS. [RL, AP, IS]. numpy array.
        """
        dk = []
        for i in range(H0.shape[0]):
            dk.append(angle2dk(
                H0[i,:], 
                meta['sizeVol'], 
                np.array(meta['sizeVol']) * np.array(meta['voxSize'])
                ))
        dk = np.stack(dk, axis=0)

        dk = torch.from_numpy(dk).float()
        mask = torch.from_numpy(mask).float()
        return cls(H0=H0, meta=meta, dk=dk, mask=mask, scaling=scaling)

    @classmethod
    def from_dk(cls, dk, mask):
        """
        Inputs:
            dk: (nori, nx, ny, nz). dipole kernel. LPS. [RL, AP, IS]. torch.Tensor.
            mask: (nx, ny, nz). brain mask. LPS. [RL, AP, IS]. torch.Tensor.
        """
        dk = dk.float()
        mask = mask.float()
        return cls(dk=dk, mask=mask)

        

    def to(self, device):
        self.dk = self.dk.to(device)
        self.mask = self.mask.to(device)
        self.device = device
        return self

    def forward(self, x):
        """
        Input:
            x: (nx, ny, nz).
        Output:
            y: (nori, nx, ny, nz).
        """
        return Phi(
            x.unsqueeze(0), 
            self.dk.unsqueeze(0), 
            self.mask.unsqueeze(0).unsqueeze(0)
            ).squeeze(0) * self.scaling

    def adjoint(self, y):
        """
        Input:
            y: (nori, nx, ny, nz).
        Output:
            x: (nx, ny, nz).
        """
        return PhiH(
            y.unsqueeze(0), 
            self.dk.unsqueeze(0), 
            self.mask.unsqueeze(0).unsqueeze(0)
            ).squeeze(0) * self.scaling

    def gram(self, x):
        """
        PhiH_Phi.
        Input:
            x: (nx, ny, nz).
        """
        return PhiH_Phi(
            x.unsqueeze(0), 
            self.dk.unsqueeze(0), 
            self.mask.unsqueeze(0).unsqueeze(0)
            ).squeeze(0) * self.scaling * self.scaling

    def pinv(self, y, epsilon=1e-6):
        """
        Pseudo-inverse of Phi.
        Input:
            y: (nori, nx, ny, nz).
            epsilon: float. l2 regularization parameter.
        Output:
            x: (nx, ny, nz).
        """
        return PhiH_Phi_plus_epsilonI_inv_PhiH(
            y.unsqueeze(0), 
            self.dk.unsqueeze(0), 
            self.mask.unsqueeze(0).unsqueeze(0),
            epsilon,
            ).squeeze(0) / self.scaling

    def add_noise(self, y, snr):
        """
        Add noise to y.
        Input:
            y: (nori, nx, ny, nz). 
            snr: float. signal to noise ratio.
        Output:
            y: (nori, nx, ny, nz). 
        """
        std = torch.sqrt(torch.mean(y ** 2) / 10 ** (snr / 10))
        noise = torch.randn_like(y) * std
        y = y + noise
        return y
        