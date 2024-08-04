import numpy as np
from scipy.sparse.linalg import cg
from scipy.ndimage import convolve
from biocard_angle import conv_kernel_rot_c0
from scipy.io import loadmat
import os

oj = os.path.join

base_dir = '/cis/home/sorenst3/my_documents/unsup_moi/data'

def gradient_mask(gradient_weighting_mode, iMag, Mask, grad, voxel_size, percentage=0.9):
    if gradient_weighting_mode == 1:
        field_noise_level = 0.01 * np.max(iMag)
        
        wG = np.abs(grad(iMag * (Mask > 0), voxel_size))
        denominator = np.sum(Mask == 1)
        numerator = np.sum(wG > field_noise_level)
        
        while (numerator / denominator) > percentage:
            field_noise_level *= 1.05
            numerator = np.sum(wG > field_noise_level)
        
        while (numerator / denominator) < percentage:
            field_noise_level *= 0.95
            numerator = np.sum(wG > field_noise_level)
        
        wG = (wG <= field_noise_level)
    
    elif gradient_weighting_mode == 2:
        # Placeholder for grayscale weighting
        pass
    
    return wG

def cdiv(Gx, voxel_size):
    cx = 0.5 * np.roll(Gx[..., 0], shift=(1, 0, 0), axis=(0, 1, 2)) - 0.5 * np.roll(Gx[..., 0], shift=(-1, 0, 0), axis=(0, 1, 2))
    cx[0, :, :] = -Gx[0, :, :, 0] - 0.5 * Gx[1, :, :, 0]
    cx[1, :, :] = Gx[0, :, :, 0] - 0.5 * Gx[2, :, :, 0]
    cx[-2, :, :] = 0.5 * Gx[-3, :, :, 0] - Gx[-1, :, :, 0]
    cx[-1, :, :] = 0.5 * Gx[-2, :, :, 0] + Gx[-1, :, :, 0]
    cx = cx / voxel_size[0]

    cy = 0.5 * np.roll(Gx[..., 1], shift=(0, 1, 0), axis=(0, 1, 2)) - 0.5 * np.roll(Gx[..., 1], shift=(0, -1, 0), axis=(0, 1, 2))
    cy[:, 0, :] = -Gx[:, 0, :, 1] - 0.5 * Gx[:, 1, :, 1]
    cy[:, 1, :] = Gx[:, 0, :, 1] - 0.5 * Gx[:, 2, :, 1]
    cy[:, -2, :] = 0.5 * Gx[:, -3, :, 1] - Gx[:, -1, :, 1]
    cy[:, -1, :] = 0.5 * Gx[:, -2, :, 1] + Gx[:, -1, :, 1]
    cy = cy / voxel_size[1]

    cz = 0.5 * np.roll(Gx[..., 2], shift=(0, 0, 1), axis=(0, 1, 2)) - 0.5 * np.roll(Gx[..., 2], shift=(0, 0, -1), axis=(0, 1, 2))
    cz[:, :, 0] = -Gx[:, :, 0, 2] - 0.5 * Gx[:, :, 1, 2]
    cz[:, :, 1] = Gx[:, :, 0, 2] - 0.5 * Gx[:, :, 2, 2]
    cz[:, :, -2] = 0.5 * Gx[:, :, -3, 2] - Gx[:, :, -1, 2]
    cz[:, :, -1] = 0.5 * Gx[:, :, -2, 2] + Gx[:, :, -1, 2]
    cz = cz / voxel_size[2]

    c = cx + cy + cz
    return c

def cgrad(x, voxel_size):
    if len(voxel_size) < 3:
        voxel_size = (1, 1, 1)

    Dx = -0.5 * np.roll(x, shift=(1, 0, 0), axis=(0, 1, 2)) + 0.5 * np.roll(x, shift=(-1, 0, 0), axis=(0, 1, 2))
    Dx[0, :, :] = -x[0, :, :] + x[1, :, :]
    Dx[-1, :, :] = x[-1, :, :] - x[-2, :, :]
    Dx = Dx / voxel_size[0]

    Dy = -0.5 * np.roll(x, shift=(0, 1, 0), axis=(0, 1, 2)) + 0.5 * np.roll(x, shift=(0, -1, 0), axis=(0, 1, 2))
    Dy[:, 0, :] = -x[:, 0, :] + x[:, 1, :]
    Dy[:, -1, :] = x[:, -1, :] - x[:, -2, :]
    Dy = Dy / voxel_size[1]

    Dz = -0.5 * np.roll(x, shift=(0, 0, 1), axis=(0, 1, 2)) + 0.5 * np.roll(x, shift=(0, 0, -1), axis=(0, 1, 2))
    Dz[:, :, 0] = -x[:, :, 0] + x[:, :, 1]
    Dz[:, :, -1] = x[:, :, -1] - x[:, :, -2]
    Dz = Dz / voxel_size[2]

    Gx = np.stack((Dx, Dy, Dz), axis=-1)
    return Gx



def delta2chi_MEDI(deltaB, params, m, mask, lambda_val=1000, merit=False, edge_per=0.9, smv_rad=0):
    if smv_rad > 0:
        N = mask.shape
        SMV = gen_SMVkernel_voxel_scaled(params, N, smv_rad)
        D = np.fft.ifftshift(convolve(np.fft.ifftn(SMV*np.fft.fftn(deltaB)), SMV)).real * mask
    else:
        D = conv_kernel_rot_c0(params, params.TAng)
        D = np.fft.ifftshift(D)

    b0 = m * np.exp(1j * deltaB)
    wG = gradient_mask(1, m, mask, params.voxSize, edge_per)

    cg_max_iter = 100
    cg_tol = 0.1

    max_iter = 10
    tol_norm_ratio = 0.1

    grad = lambda x: cgrad(x, params.voxSize)
    div = lambda x: cdiv(x, params.voxSize)

    iter = 0
    x = np.zeros(params.sizeVol)
    res_norm_ratio = np.inf
    cost_data_history = np.zeros(max_iter)
    cost_reg_history = np.zeros(max_iter)

    e = 0.000001

    while res_norm_ratio > tol_norm_ratio and iter < max_iter:
        iter += 1
        Vr = 1.0 / np.sqrt(np.abs(wG * grad(x)) ** 2 + e)

        w = m * np.exp(1j * np.fft.ifftn(D * np.fft.fftn(x)).real)

        reg = lambda dx: div(wG * (Vr * (wG * grad(dx))))
        fidelity = lambda dx: 2 * lambda_val * np.real(np.fft.ifftn(D * np.fft.fftn(np.conj(w) * w * np.fft.ifftn(D * np.fft.fftn(dx)))))

        A = lambda dx: reg(dx) + fidelity(dx)
        b = reg(x) + 2 * lambda_val * np.real(np.fft.ifftn(D * np.fft.fftn(np.conj(w) * np.conj(1j) * (w - b0))))

        dx, _ = cg(A, -b, tol=cg_tol, maxiter=cg_max_iter)
        res_norm_ratio = np.linalg.norm(dx) / np.linalg.norm(x)
        x += dx

        wres = m * np.exp(1j * np.fft.ifftn(D * np.fft.fftn(x)).real) - b0
        cost_data_history[iter - 1] = np.linalg.norm(wres)
        cost = np.abs(wG * grad(x))
        cost_reg_history[iter - 1] = np.sum(cost)

        if merit:
            wres -= np.mean(wres[mask == 1])
            a = wres[mask == 1]
            factor = np.std(np.abs(a)) * 6
            wres = np.abs(wres) / factor
            wres[wres < 1] = 1
            m /= wres ** 2
            b0 = m * np.exp(1j * deltaB)

        print(f'iter: {iter}; res_norm_ratio: {res_norm_ratio:.4f}; cost_L2: {cost_data_history[iter - 1]:.4f}; cost_Reg: {cost_reg_history[iter - 1]:.4f}.')

    x *= mask
    return x, cost_reg_history, cost_data_history


def gen_SMVkernel_voxel_scaled(params, N, smv_rad):
    Y, X, Z = np.meshgrid(np.arange(-N[1] // 2, np.ceil(N[1] / 2)), np.arange(-N[0] // 2, np.ceil(N[0] / 2)),
                           np.arange(-N[2] // 2, np.ceil(N[2] / 2)))

    X *= params.voxSize[0]
    Y *= params.voxSize[1]
    Z *= params.voxSize[2]

    smv = (X ** 2 + Y ** 2 + Z ** 2) <= smv_rad ** 2
    smv = smv / np.sum(smv)
    smv_kernel = np.zeros_like(X)
    smv_kernel[smv_kernel.shape[0] // 2, smv_kernel.shape[1] // 2, smv_kernel.shape[2] // 2] = 1
    smv_kernel -= smv

    SMV = np.fft.fftn(np.fft.ifftshift(smv_kernel))
    return SMV

if __name__ == '__main__':
    sub = 1
    ori = 1
    mat_filename = oj(base_dir, 'Sub{0:04d}'.format(sub), 'ori{0}'.format(ori), 
                      'Sub{0:04d}_ori{1}_dipole.mat'.format(sub, ori))                              
    mat = loadmat(mat_filename)
    # deltaB - the normalized phase measurement
    params = mat['Params']
    # m - 
    mask = mat['maskErode']

    delta2chi_MEDI(deltaB, params, m, mask)