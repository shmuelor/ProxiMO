
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


class ANI_loss(nn.Module):
	def __init__(self, device):
		super().__init__()
		self.cuda_device = device

	def forward(self, input_data, target, mask, root_dir):

		gt_mean = torch.from_numpy(np.load(root_dir + 'partition/partition_data6_list/train_gt_mean.npy')[:, np.newaxis, np.newaxis, np.newaxis]).float()
		gt_std = torch.from_numpy(np.load(root_dir + 'partition/partition_data6_list/train_gt_mean.npy')[:, np.newaxis, np.newaxis, np.newaxis]).float()

		cpu_device = torch.device('cpu')
		
		(batch, _, H, W, D) = input_data.size()

		new_input = ((input_data * gt_std.to(target.device, dtype=torch.float)) + gt_mean.to(target.device, dtype=torch.float)) * mask
		new_target = ((target * gt_std.to(target.device, dtype=torch.float)) + gt_mean.to(target.device, dtype=torch.float)) * mask

		new_input = transform_matrix(new_input).to(cpu_device)
		new_target = transform_matrix(new_target).to(cpu_device)

		input_eigval, _ = torch.symeig(new_input[mask[:, 0, :, :, :] == 1], True)
		target_eigval, _ = torch.symeig(new_target[mask[:, 0, :, :, :] == 1], True)

		input_eigval = input_eigval.to(self.cuda_device)
		target_eigval = target_eigval.to(self.cuda_device)
		
		input_ani = input_eigval[:, 2] - 0.5 * (input_eigval[:, 0] + input_eigval[:, 1])
		target_ani = target_eigval[:, 2] = 0.5 * (target_eigval[:, 0] + target_eigval[:, 1]) 

		return F.l1_loss(input_ani, target_ani)


def transform_matrix(tensor_data):
	
	# tensor as a 4D file in this order: Dxx,Dxy,Dxz,Dyy,Dyz,Dzz
	(B, channel, H, W, D) = tensor_data.size()
	
	matrix_data = torch.zeros((B, H, W, D, 3, 3)).to(tensor_data.device, dtype=torch.float)
	#matrix_data = tensor_data.new_tensor(zero_data)
	
	matrix_data[:, :, :, :, 0, 0] = tensor_data[:, 0, :, :, :]
	matrix_data[:, :, :, :, 0, 1] = tensor_data[:, 1, :, :, :]
	matrix_data[:, :, :, :, 0, 2] = tensor_data[:, 2, :, :, :]
	matrix_data[:, :, :, :, 1, 0] = tensor_data[:, 1, :, :, :]
	matrix_data[:, :, :, :, 1, 1] = tensor_data[:, 3, :, :, :]
	matrix_data[:, :, :, :, 1, 2] = tensor_data[:, 4, :, :, :]
	matrix_data[:, :, :, :, 2, 0] = tensor_data[:, 2, :, :, :]
	matrix_data[:, :, :, :, 2, 1] = tensor_data[:, 4, :, :, :]
	matrix_data[:, :, :, :, 2, 2] = tensor_data[:, 5, :, :, :]

	return matrix_data
