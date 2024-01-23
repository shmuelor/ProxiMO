import numpy as np
import torch

class StiOperatorToolkit(object):
    """
    Functions related to forward operator in susceptibility tensor imaging (sti)
    """
    def __init__(self):
        pass

    @staticmethod
    def angle2dk(H0, sizeVol, fov):
        """
        From the direction of magnetic field in subject frame of reference (H0)
        to dipole kernel.
        
        Inputs:
            H0: Unit vector, direction of magnetic field in [RL, AP, IS] (LPS orientation). 
                RL: subject right-left. AP: Anterior-Posterior. IS: Inferior-Superior.
            sizeVol: size of dipole kernel. [AP, RL, IS] (PLS orientation).
            fov: field of view in mm. [AP, RL, IS] (PLS orientation).
        Return:
            dipole_tensor: dipole kernel. PLS.
                           [AP, RL, IS, 6]. 
                           6 channels are in order: xx, xy, xz, yy, yz, zz. x: RL, y: AP, z: IS.
        """
        Nx = sizeVol[1]
        Ny = sizeVol[0]
        Nz = sizeVol[2]

        dkx = 1/fov[1]
        dky = 1/fov[0]
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

        kx = kx.astype('float32')
        ky = ky.astype('float32')
        kz = kz.astype('float32')

        KX_Grid, KY_Grid, KZ_Grid = np.meshgrid(kx, ky, kz) # mesh in k space
        KSq = KX_Grid ** 2 + KY_Grid ** 2 + KZ_Grid ** 2          # k^2

        hx, hy, hz = H0

        B0 = (hx*KX_Grid + hy*KY_Grid + hz*KZ_Grid)/KSq
        d1 = (1/3)*(hx*hx) - B0*KX_Grid*hx
        d2 = (2/3)*(hx*hy) - B0*(KX_Grid*hy + KY_Grid*hx)
        d3 = (2/3)*(hx*hz) - B0*(KX_Grid*hz + KZ_Grid*hx)
        d4 = (1/3)*(hy*hy) - B0*KY_Grid*hy
        d5 = (2/3)*(hy*hz) - B0*(KY_Grid*hz + KZ_Grid*hy)
        d6 = (1/3)*(hz*hz) - B0*KZ_Grid*hz

        d1 = np.fft.ifftshift(d1)
        d2 = np.fft.ifftshift(d2)
        d3 = np.fft.ifftshift(d3)
        d4 = np.fft.ifftshift(d4)
        d5 = np.fft.ifftshift(d5)
        d6 = np.fft.ifftshift(d6)

        d1[np.isnan(d1)] = 0
        d2[np.isnan(d2)] = 0
        d3[np.isnan(d3)] = 0
        d4[np.isnan(d4)] = 0
        d5[np.isnan(d5)] = 0
        d6[np.isnan(d6)] = 0

        dipole_tensor = np.zeros([sizeVol[0], sizeVol[1], sizeVol[2], 6], dtype='float32')

        dipole_tensor[:, :, :, 0] = d1
        dipole_tensor[:, :, :, 1] = d2
        dipole_tensor[:, :, :, 2] = d3
        dipole_tensor[:, :, :, 3] = d4
        dipole_tensor[:, :, :, 4] = d5
        dipole_tensor[:, :, :, 5] = d6

        dipole_tensor = dipole_tensor.astype('float32')

        return dipole_tensor
    
    @classmethod
    def angle2dk_LPS(cls, H0, sizeVol, fov):
        """
        From the direction of magnetic field in subject frame of reference (H0)
        to dipole kernel.
        Assumes everything in LPS orientation.
        
        Inputs:
            H0: Unit vector, direction of magnetic field. LPS. 
            sizeVol: size of dipole kernel. LPS.
            fov: field of view in mm. LPS.
        Return:
            dipole_tensor: dipole kernel. LPS.
                           [RL, AP, IS, 6]. 
                           6 channels are in order: xx, xy, xz, yy, yz, zz. x: RL, y: AP, z: IS.
        """
        dipole_tensor = cls.angle2dk(H0, [sizeVol[1],sizeVol[0],sizeVol[2]], [fov[1],fov[0],fov[2]]) # dipole kernel in PLS
        dipole_tensor = np.transpose(dipole_tensor, (1,0,2,3))
        return dipole_tensor
        
    @staticmethod
    def Phi(x, dk, m):
        """
        Implementation of \Phi, where \Phi is forward operator of STI.
        y = \Phi * x = mask * F^-1 * D * F * x.
        All inputs and outputs are torch.Tensor.

        Input:
            x: batch, chi(6), w, h, d. 
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
    
    @staticmethod
    def PhiH(x, dk):
        """
        Implementation of \Phi^H, where \Phi is forward operator of STI, ^H is Hermitian transpose.
        y = \Phi^H * x = F^-1 * D^H * F * mask^H * x.
        Note: Since x is zero outside the brain mask, we omit the mask multiplication step before FFT.

        Input:
            x: batch, 1, orientations, w, h, d
            dk: dipole kernel. batch, chi(6), orientations, w, h, d, 1
        Return:
            batch, chi(6), w, h, d. 
        """
        x = _rfft(x)
        x = dk * x
        x = torch.sum(x, dim=2)
        x = _ifft(x)[:, :, :, :, :, 0]
        return x

    @staticmethod
    def PhiH_Phi(x, dk, m):
        """
        Implementation of \Phi^H \Phi, where \Phi is forward operator of STI, ^H is Hermitian transpose.
        
        Input:
            x: batch, chi(6), w, h, d
            dk: dipole kernel. batch, chi(6), orientations, w, h, d, 1
            m: brain mask. batch, 1, orientations, w, h, d
        Return:
            batch, chi(6), w, h, d. 
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
    
    
#####################
# Fourier transforms
#####################

def _rfft(x):
    """
    Fourier Transform of real signal.
    
    Input:
        x: [..., w, h, d], real.
    Return:
        [..., w, h, d, 2]. The last dimension saves real and imaginary parts of the result separately.
    """
    try:
        # torch==1.2.0
        x = torch.rfft(x, 3, normalized=True, onesided=False)
    except:
        # torch==1.10.0
        x = torch.fft.fftn(x, dim=(-3, -2, -1), norm='ortho')
        x = torch.stack((x.real, x.imag), dim=-1)
    return x

def _ifft(x):
    """
    Inverse Fourier Transform.
    
    Input:
        x: [..., w, h, d, 2]. The last dimension saves real and imaginary parts.
    Return:
        [..., w, h, d, 2]. The last dimension saves real and imaginary parts.
    """
    try:
        # torch==1.2.0
        x = torch.ifft(x, 3, normalized=True)
    except:
        # torch==1.10.0
        x = torch.view_as_complex(x)
        x = torch.fft.ifftn(x, dim=(-3, -2, -1), norm='ortho')
        x = torch.stack((x.real, x.imag), dim=-1)
    return x


