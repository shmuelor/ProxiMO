import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def qsm_psnr(gt, input_data, mask, roi=True, subtract_mean=False):

	max_val = 0.60784858
	min_val = -0.8570962

	if subtract_mean:
		gt = gt - np.mean(gt)
		input_data = input_data - np.mean(input_data)
	
	mod_input = np.copy(input_data)
	mod_input[mod_input < min_val] = min_val
	mod_input[mod_input > max_val] = max_val
	
	if roi:
		psnr_value = psnr(gt[mask==1], mod_input[mask==1], data_range=max_val - min_val)
	else:
		psnr_value = psnr(gt, mod_input, data_range=max_val - min_val)
	
	return psnr_value

def qsm_ssim(gt, input_data, subtract_mean=False):

	max_val = 0.60784858
	min_val = -0.8570962

	if subtract_mean:
		gt = gt - np.mean(gt)
		input_data = input_data - np.mean(input_data)

	mod_input = np.copy(input_data)
	
	mod_input[mod_input < min_val] = min_val
	mod_input[mod_input > max_val] = max_val

	new_gt = (gt - min_val) / (max_val - min_val)
	new_input = (mod_input - min_val) / (max_val - min_val)
	
	ssim_value = ssim(new_gt, new_input, multichannel=True, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

	return ssim_value