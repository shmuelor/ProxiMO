import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim

class StiEvaluationToolkit(object):
    def __init__(self):
        pass

    @staticmethod
    def tensor2misc(sti):
        """
        Convert tensor image to miscellaneous
        
        Input:
            sti: [w,h,d,6]
        Return:
            tuple of (
                L: eigenvalues, [w,h,d,3], descending order
                V: eigenvectors, [w,h,d,3,3], last dim is index of each eigenvector
                avg: mean of eigenvalues, [w,h,d], ie, mean magnetic susceptibility, MMS
                ani: magnetic susceptibility anisotropy (MSA), [w,h,d]
                V1: principal evect, [w,h,d,3]
                modpev: modulated pev, [w,h,d,3]
            )
        """
        matrix_data = transform_matrix(sti)
        if np.isnan(matrix_data).any():
            print('NaN encountered when calculating eigenvalues!')
            L = np.zeros(matrix_data.shape[0:3]+(3,)) * np.NaN
            V = np.zeros(matrix_data.shape[0:3]+(3,3)) * np.NaN
        else:
            L, V = np.linalg.eigh(matrix_data)
        # change ascending to descending order
        L = np.flip(L, axis=3)
        V = np.flip(V, axis=4)
        avg = (L[...,0] + L[...,1] + L[...,2]) / 3
        ani = L[...,0] - (L[...,1] + L[...,2]) / 2
        V1 = V[...,0]
        modpev = V1 * ani[:,:,:,None]
        return L, V, avg, ani, V1, modpev

    @staticmethod
    def psnr_sti(a, b, mask, data_min=-0.8574115, data_max=0.63124019):
        """
        psnr with b as ground-truth
        Input:
            a, b: (w,h,d,6)
            mask: (w,h,d)
            data_min, data_max: scalars, min and max value of all data
        Output:
            scalar, psnr value
        """
        assert a.shape[-1] == 6 and b.shape[-1] == 6
        a = np.clip(a, data_min, data_max)
        data_range = data_max - data_min # max{all data} - min{all data}
        max_sig_power = (data_range)**2
        noise_power = np.average(np.mean((a-b)**2,axis=-1), weights=mask)
        psnr = 10*np.log10(max_sig_power/noise_power)
        return psnr

    @staticmethod
    def ssim_sti(gt, pred, data_max, data_min):
        """
        ssim for sti image.
        Input:
            gt, pred: (w,h,d,6)
            data_max, data_min: (6,) maximum and minimum of STI data per channel
        Output:
            scalar, ssim value
        """
        pred = np.copy(pred)
        
        # clip to data range
        for i in range(6):
            pred[:, :, :, i][pred[:, :, :, i] < data_min[i]] = data_min[i]
            pred[:, :, :, i][pred[:, :, :, i] > data_max[i]] = data_max[i]

        new_gt = (gt - data_min) / (data_max - data_min)
        new_pred = (pred - data_min) / (data_max - data_min)
        
        ssim_value = skimage_ssim(new_gt, new_pred, channel_axis=-1, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

        return ssim_value

    @staticmethod
    def vector_error(a, b):
        """
        vector error in terms of |cosine similarity|
        Input:
            a, b: (..., 3) unweighted pev map
        Output:
            (...): vector error between a, b at each voxel
        """
        assert a.shape[-1] == 3 and b.shape[-1] == 3
        dot_prod = np.sum(a*b, axis=-1) # dot product
        norm_a = np.sqrt(np.sum(a**2, axis=-1))
        norm_b = np.sqrt(np.sum(b**2, axis=-1))
        cos_sim = dot_prod / (norm_a*norm_b)
        err = 1 - np.abs(cos_sim)
        return err

    @classmethod
    def evec_cos_sim_err(cls, a, b, mask, gt_ani, ani_thr=0.015):
        """
        eigenvector cosine similarity error, ECSE
        Input:
            a, b: (w,h,d,3) unweighted pev map
            mask: (w,h,d) brain mask
            gt_ani: (w,h,d) ground-truth anisotropy
            ani_thr: threshold for anisotropy. compute average only in region of anisotropy > thr.
        Output:
            scalar, ECSE
        """
        vecerr_map = cls.vector_error(a, b)
        ECSE = np.mean(vecerr_map[(mask*(gt_ani>ani_thr))==1])
        return ECSE

    @staticmethod
    def wpsnr_sti(a, weight_a, b, weight_b, mask, data_max=0.08, data_min=0):
        """
        wpsnr with b as ground-truth. psnr of modulated pev maps.
        Input:
            a, b: (w,h,d,3) unweighted pev maps
            weight_a, weight_b: (w,h,d) anisotropy maps
            mask: (w,h,d)
            data_max, data_min: scalars, min and max of all data (modulated pev maps)
        Output:
            scalar
        """
        assert a.shape[-1] == 3 and b.shape[-1] == 3
        
        a = np.abs(a)
        b = np.abs(b)
        
        mod_a = a * weight_a[:,:,:,np.newaxis]
        mod_b = b * weight_b[:,:,:,np.newaxis]
        
        mod_a = np.clip(mod_a, data_min, data_max)
        
        data_range = data_max - data_min # max{all data} - min{all data}
        max_sig_power = (data_range)**2
        noise_power = np.average(np.mean((mod_a-mod_b)**2,axis=-1), weights=mask)
        psnr = 10*np.log10(max_sig_power/noise_power)
        return psnr

    @staticmethod
    def mse_sti(pred, gt, mask):
        """
        pred, gt: (w,h,d,6)
        mask: (w,h,d)
        """
        gt = gt[mask==1]
        pred = pred[mask==1]
        mse = np.mean(np.square(gt - pred))

        return mse

def transform_matrix(x):
    """
    Transform 6-channel tensor image into 3x3 matrix form.
    Input:
        x: (..., 6), 6-channel tensor image in order: chi-[11,12,13,22,23,33]
    Output:
        (..., 3, 3): tensor image in matrix form
    """
    
    assert x.shape[-1] == 6
    out = np.zeros(x.shape[:-1]+(3, 3)) # [...,3,3]
    
    out[..., 0, 0] = x[..., 0]
    out[..., 0, 1] = x[..., 1]
    out[..., 0, 2] = x[..., 2]
    out[..., 1, 1] = x[..., 3]
    out[..., 1, 2] = x[..., 4]
    out[..., 2, 2] = x[..., 5]
    out[..., 1, 0] = x[..., 1]
    out[..., 2, 0] = x[..., 2]
    out[..., 2, 1] = x[..., 4]
    
    return out