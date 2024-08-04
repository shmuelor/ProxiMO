"""Numpy implementation of TKD reconstruction for QSM."""
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import nibabel as nib

import sti_operator_toolkit as stot
import qsm_operator_toolkit as qot


def tkd_D(delta, D, thresh):
    """TKD reconstruction for QSM, using dipole kernel.
    Parameters
    ----------
    delta : numpy.ndarray
        Local field perturbation in ppm. Shape: (nx, ny, nz).
    D : numpy.ndarray
        Dipole kernel. Shape: (nx, ny, nz). Zero-frequency is at **center** of the spectrum.
    thresh : float
        Threshold for TKD.
    Returns
    -------
    chi : numpy.ndarray
        Susceptibility map in ppm. Shape: (nx, ny, nz).
    """
    # shift zero-frequency to **boundary** of the spectrum
    D = ifftshift(D)

    # Perform FFT for delta
    delta_k = fftn(delta)

    # Create a boolean mask for elements in D less than thresh
    IndIll = np.abs(D) < thresh

    # Initialize Dinv with 1/D
    Dinv = 1.0 / D

    # Update Dinv for elements where IndIll is True
    Dinv[IndIll] = np.sign(D[IndIll]) * (1 / thresh)

    # Perform IFFT for delta_k multiplied by Dinv
    chi = np.real(ifftn(delta_k * Dinv))

    ## Correction
    ## Ref: Scheweser, 2012, MRM, toward online QSM.
    # Perform IFFT for D multiplied by Dinv
    psf = np.real(ifftn(D * Dinv))

    # Extract the first element of psf
    c = psf[0, 0, 0]
    print("correcting factor =", c)

    # Normalize chi by dividing it by c
    chi = chi / c

    return chi


# Function for TKD reconstruction for QSM, using dipole kernel or (B0 direction, fov).
def tkd(delta, thresh, D=None, b0dir=None, voxel_size=None):
    """TKD reconstruction for QSM, using dipole kernel or (B0 direction, voxel size).
    Either D or (b0dir, voxel_size) should be provided.

    Parameters
    ----------
    delta : numpy.ndarray
        Local field perturbation in ppm. Shape: (nx, ny, nz).
    thresh : float
        Threshold for TKD.
    D : numpy.ndarray, optional
        Dipole kernel. Shape: (nx, ny, nz). Zero-frequency is at **center** of the spectrum. If None, calculated based on b0dir and voxel_size.
    b0dir : numpy.ndarray, optional
        B0 direction. Shape: (3, ). Required if D is not provided.
    voxel_size : numpy.ndarray, optional
        Voxel size in mm. Shape: (3, ). Required if D is not provided.

    Returns
    -------
    chi : numpy.ndarray
        Susceptibility map in ppm. Shape: (nx, ny, nz).

    Raises
    ------
    ValueError: If both D and (b0dir, voxel_size) are None.

    Example
    -------
    >>> result = tkd(delta, thresh, D=D)
    >>> result = tkd(delta, thresh, b0dir=b0dir, voxel_size=voxel_size)
    """
    if D is None:
        # Either D or (b0dir, voxel_size) should be provided.
        assert b0dir is not None and voxel_size is not None
        # Convert b0dir to dipole kernel
        # D = qot.angle2dk(b0dir, delta.shape, voxel_size)
        Dsti = stot.angle2dk(b0dir, delta.shape, voxel_size)
        D = Dsti[:, :, :, 0] + Dsti[:, :, :, 3] + Dsti[:, :, :, 5]

    # save D as mat
    # import scipy.io as sio

    # sio.savemat("D.mat", {"D": D})
    # print(D.shape)

    # Perform TKD reconstruction
    chi = tkd_D(delta, D, thresh)

    return chi
