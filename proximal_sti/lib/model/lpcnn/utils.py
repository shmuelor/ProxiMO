import torch
import torch.nn.functional as F

def PhiH_Phi(x, dk, m):
    """
    Implementation of \Phi^H \Phi, where \Phi is forward model, ^H is Hermitian transpose
    Input:
        x: batch, chi(6), w, h, d
        dk: dipole kernel. batch, chi(6), orientations, w, h, d, 1
        m: brain mask. batch, 1, orientations, w, h, d
    """
    x = _rfft(x)
    x = x.unsqueeze(2)
    x = dk * x
    x = torch.sum(x, 1, keepdim=True)
    x = _ifft(x)[:, :, :, :, :, :, 0]
    x = x * m
    x = _rfft(x)
    x = dk * x
    x = torch.sum(x, 2)
    x = _ifft(x)[:, :, :, :, :, 0]
    return x


def PhiH(x, dk):
    """
    Implementation of \Phi^H, where \Phi is forward model, ^H is Hermitian transpose.
    Note: Since x is zero outside the brain mask, we omit the mask multiplication step before FFT.
    
    Input:
        x: batch, 1, orientations, w, h, d
        dk: dipole kernel. batch, chi(6), orientations, w, h, d, 1
    """
    x = _rfft(x)
    x = dk * x
#     x *= dk
    x = torch.sum(x, dim=2)
    x = _ifft(x)[:, :, :, :, :, 0]
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
    x = _rfft(x)
    x = x.unsqueeze(2)
    x = dk * x
    x = torch.sum(x, 1, keepdim=True)
    x = _ifft(x)[:, :, :, :, :, :, 0]
    x = x * m
    return x


def _rfft(x):
    try:
        # torch==1.2.0
        x = torch.rfft(x, 3, normalized=True, onesided=False)
    except:
        # torch==1.10.0
        x = torch.fft.fftn(x, dim=(-3, -2, -1), norm='ortho')
        x = torch.stack((x.real, x.imag), dim=-1)
    return x


def _ifft(x):
    try:
        # torch==1.2.0
        x = torch.ifft(x, 3, normalized=True)
    except:
        # torch==1.10.0
        x = torch.view_as_complex(x)
        x = torch.fft.ifftn(x, dim=(-3, -2, -1), norm='ortho')
        x = torch.stack((x.real, x.imag), dim=-1)
    return x


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

