"""Functions for QSM operator"""
import numpy as np


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
        Dipole kernel. Shape: (nx, ny, nz).
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

    dk = 1 / 3 - (hx * KX_Grid + hy * KY_Grid + hz * KZ_Grid) ** 2 / KSq

    dk[np.isnan(dk)] = 0

    return dk
