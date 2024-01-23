
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


class DECOM_loss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, input_data, target, mask, root_dir):

		mask = mask[:, :, 0, :, :, :].bool()
		mask_6 = mask.expand(-1, 6, -1, -1, -1)

		#gt_mean = torch.from_numpy(np.load(root_dir + 'partition/partition_data6_list/train_gt_mean.npy')[:, np.newaxis, np.newaxis, np.newaxis]).float()
		#gt_std = torch.from_numpy(np.load(root_dir + 'partition/partition_data6_list/train_gt_mean.npy')[:, np.newaxis, np.newaxis, np.newaxis]).float()

		(batch, _, H, W, D) = input_data.size()

		#new_input = ((input_data * gt_std.to(target.device, dtype=torch.float)) + gt_mean.to(target.device, dtype=torch.float))
		#new_target = ((target * gt_std.to(target.device, dtype=torch.float)) + gt_mean.to(target.device, dtype=torch.float))

		input_iso = (input_data[:, 0, :, :, :] + input_data[:, 3, :, :, :] + input_data[:, 5, :, :, :]) / 3
		input_iso = input_iso.unsqueeze(1)
		zero_iso = torch.zeros_like(input_iso)
		expand_input_iso = torch.cat([input_iso, zero_iso, zero_iso, input_iso, zero_iso, input_iso], dim=1)
		input_aniso = input_data - expand_input_iso.detach()


		target_iso = (target[:, 0, :, :, :] + target[:, 3, :, :, :] + target[:, 5, :, :, :]) / 3
		target_iso = target_iso.unsqueeze(1)
		expand_target_iso = torch.cat([target_iso, zero_iso, zero_iso, target_iso, zero_iso, target_iso], dim=1)
		target_aniso = target - expand_target_iso

		iso_loss = F.l1_loss(input_iso[mask], target_iso[mask])
		aniso_loss = F.l1_loss(input_aniso[mask_6], target_aniso[mask_6])


		return iso_loss, aniso_loss


