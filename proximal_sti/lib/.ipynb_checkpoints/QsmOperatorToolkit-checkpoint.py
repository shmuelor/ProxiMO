import numpy as np
import torch

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
        m: mask. batch, orientations, w, h, d
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
        m: mask. batch, orientations, w, h, d
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
        m: brain mask. batch, orientations, w, h, d
    Return:
        batch, w, h, d. 
    """
    x = _rfft(x)
    x = x.unsqueeze(1)
    x = dk.unsqueeze(-1) * x
    x = _ifft(x)[:, :, :, :, :, 0]
    x = x * m
    x = _rfft(x)
    x = dk.unsqueeze(-1) * x
    x = torch.sum(x, dim=1)
    x = _ifft(x)[:, :, :, :, 0]
    return x