
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


class WPEV_loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cuda_device = device

    def forward(self, input_data, target, mask, gt_mean, gt_std, ani_mask):

#         gt_mean = torch.from_numpy(np.load(root_dir + 'partition/partition_data6_list/train_gt_mean.npy')[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]).float()
#         gt_std = torch.from_numpy(np.load(root_dir + 'partition/partition_data6_list/train_gt_std.npy')[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]).float()

        cpu_device = torch.device('cpu')
#         cpu_device = torch.device('cuda:0')
#         torch.cuda.set_device("cuda:0")
        
        (batch, _, H, W, D) = input_data.size()

#         print(input_data.shape, gt_std.shape, gt_mean.shape, mask.shape)
        new_input = ((input_data * gt_std.to(target.device, dtype=torch.float)) + gt_mean.to(target.device, dtype=torch.float)) * mask[:, :, 0, :, :, :]
        new_target = ((target * gt_std.to(target.device, dtype=torch.float)) + gt_mean.to(target.device, dtype=torch.float)) * mask[:, :, 0, :, :, :]

        #new_input = input_data
        #new_target = target
        
        new_input = transform_matrix(new_input).to(cpu_device)
        new_target = transform_matrix(new_target).to(cpu_device)

#         print(new_input.shape, mask.shape, ani_mask.shape)
        bool_mask = mask > 0
#         bool_mask = mask[:, 0, 0, :, :, :] == 1
        
        
        input_eigval, input_eigvec = torch.symeig(new_input[bool_mask[:,0,0,:,:,:]], True)
        target_eigval, target_eigvec = torch.symeig(new_target[bool_mask[:,0,0,:,:,:]], True)
#         input_eigval, input_eigvec = torch.linalg.eigh(new_input[bool_mask[:,0,0,:,:,:]])
#         target_eigval, target_eigvec = torch.linalg.eigh(new_target[bool_mask[:,0,0,:,:,:]])
        
    
        input_eigval = input_eigval.to(self.cuda_device)
        target_eigval = target_eigval.to(self.cuda_device)

        #input_eigvec = input_eigvec[:, :, 2].to(self.cuda_device)
        #target_eigvec = target_eigvec[:, :, 2].to(self.cuda_device) 

        #cos = nn.CosineSimilarity(dim=1, eps=1e-08)

        #return 1 - (torch.mean(cos(input_eigvec, target_eigvec)))

        input_eigvec = input_eigvec.to(self.cuda_device)
        target_eigvec = target_eigvec.to(self.cuda_device)

#         loss = 1 - torch.mean(torch.abs(torch.sum(input_eigvec[:, :, 2] * target_eigvec[:, :, 2], dim=1)))
        #loss = 3 - (torch.mean(torch.abs(torch.sum(input_eigvec[:, :, 2] * target_eigvec[:, :, 2], dim=1))) + torch.mean(torch.abs(torch.sum(input_eigvec[:, :, 1] * target_eigvec[:, :, 1], dim=1))) + torch.mean(torch.abs(torch.sum(input_eigvec[:, :, 0] * target_eigvec[:, :, 0], dim=1))))
        
        input_ani = input_eigval[:, 2] - (input_eigval[:, 1] + input_eigval[:, 0]) / 2
        target_ani = target_eigval[:, 2] - (target_eigval[:, 1] + target_eigval[:, 0]) / 2
        
        input_wpev = input_ani.unsqueeze(1) * input_eigvec[:,:,2]
        target_wpev = target_ani.unsqueeze(1) * target_eigvec[:,:,2]

        loss = F.l1_loss(input_wpev, target_wpev)
        
        return loss

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
