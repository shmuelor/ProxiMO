import numpy as np

def angle2dk(H0, sizeVol, fov):
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

#     kx = np.linspace(-Nx/2, Nx/2-1, Nx) * dkx
#     ky = np.linspace(-Ny/2, Ny/2-1, Ny) * dky
#     kz = np.linspace(-Nz/2, Nz/2-1, Nz) * dkz
    
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
