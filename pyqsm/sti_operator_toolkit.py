import numpy as np
import torch


def angle2dk(b0dir, size_vol, voxel_size):
    """Convert B0 direction to dipole kernel.
    Parameters
    ----------
    b0dir : numpy.ndarray
        B0 direction. Shape: (3, ).
    size_vol : numpy.ndarray
        Size of the volume. Shape: (3, ).
    voxel_size : numpy.ndarray
        Voxel size in mm. Shape: (3, ).
    Returns
    -------
    dk : numpy.ndarray
        Dipole kernel. Shape: (nx, ny, nz, 6).
        Zero-frequency is at **center** of the spectrum.
    """
    Nx = size_vol[0]
    Ny = size_vol[1]
    Nz = size_vol[2]

    fov = np.array(voxel_size) * np.array(size_vol)

    dkx = 1 / fov[0]
    dky = 1 / fov[1]
    dkz = 1 / fov[2]

    # convolution kernel
    def Ni2linspace(Ni):
        if Ni % 2 == 0:
            pts = np.linspace(-Ni / 2, Ni / 2 - 1, Ni)
        else:
            pts = np.linspace(-(Ni - 1) / 2, (Ni - 1) / 2, Ni)
        return pts

    kx = Ni2linspace(Nx) * dkx
    ky = Ni2linspace(Ny) * dky
    kz = Ni2linspace(Nz) * dkz

    kx = kx.astype("float32")
    ky = ky.astype("float32")
    kz = kz.astype("float32")

    KX_Grid, KY_Grid, KZ_Grid = np.meshgrid(
        kx, ky, kz, indexing="ij"
    )  # mesh in k space
    KSq = KX_Grid**2 + KY_Grid**2 + KZ_Grid**2  # k^2

    hx, hy, hz = b0dir

    B0 = (hx * KX_Grid + hy * KY_Grid + hz * KZ_Grid) / KSq
    d1 = (1 / 3) * (hx * hx) - B0 * KX_Grid * hx
    d2 = (2 / 3) * (hx * hy) - B0 * (KX_Grid * hy + KY_Grid * hx)
    d3 = (2 / 3) * (hx * hz) - B0 * (KX_Grid * hz + KZ_Grid * hx)
    d4 = (1 / 3) * (hy * hy) - B0 * KY_Grid * hy
    d5 = (2 / 3) * (hy * hz) - B0 * (KY_Grid * hz + KZ_Grid * hy)
    d6 = (1 / 3) * (hz * hz) - B0 * KZ_Grid * hz

    d1[np.isnan(d1)] = 0
    d2[np.isnan(d2)] = 0
    d3[np.isnan(d3)] = 0
    d4[np.isnan(d4)] = 0
    d5[np.isnan(d5)] = 0
    d6[np.isnan(d6)] = 0

    dk = np.zeros([size_vol[0], size_vol[1], size_vol[2], 6], dtype="float32")

    dk[:, :, :, 0] = d1
    dk[:, :, :, 1] = d2
    dk[:, :, :, 2] = d3
    dk[:, :, :, 3] = d4
    dk[:, :, :, 4] = d5
    dk[:, :, :, 5] = d6

    dk = dk.astype("float32")

    return dk
