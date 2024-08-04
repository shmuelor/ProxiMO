# unsupervised learning with multiple operators

import os

oj = os.path.join
import sys

import nibabel as nib
import numpy as np
# import scipy.io
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch

# print(torch.cuda.device_count())
print(torch.cuda.current_device())
from tensorboardX import SummaryWriter

sys.path.append('.')
sys.path.append('./proximal_sti')
from unet3d_test import evaluate

import torch.nn as nn
import torch.optim as optim
from data import get_loader
# from pdata import get_loader
from loss import CrossLoss, MCLoss
from lpcnn import LPCNN as LPCNN_model
from proximal_sti.lib.QsmEvaluationToolkit import qsm_psnr
from utils import (construct_A, denormalize, get_dk_shape, sample_A,
                   sample_snr)

from unet3d_adain import UNet3DAdaINCodeGen
from unet3d_losses import gradient_difference_loss, total_variation_loss

lpcnn = False
train_on_sim = False
is_aug = True # TODO - they did True
scaling_fac = 1.
loss_type = 'l2'
lr = 1e-4
batch_size = 1
# test_sep = 'whole'
norm = True
unet = False
# mc_fac = 5e3

x_avg_max = 0.6
y_avg_max = 8.5e-5
beta = 0.5
gamma = 0.8

device = 'cuda'
save_dir = 'save_dir/unet_adain_noltv_bio_aug_zf'
os.makedirs(save_dir, exist_ok=True)
use_pinv = False

# model.load_state_dict(torch.load(oj(save_dir,'model_best.pth')))

ds1_train = [1,2,4,5,8,9]
ds1_val = [3, 6]
# ds2 = np.arange(101, 113)
biocard = np.concatenate((np.arange(1001, 1101), np.arange(2001, 2101), np.arange(3001, 3101),
                            np.arange(4001, 4101), np.arange(5001, 5100)))
# biocard_all = np.concatenate((biocard, np.arange(6001, 6024)))

train_loader = get_loader('train', train_w_cosmos=[], train_wo_cosmos=np.concatenate((ds1_train, biocard)), batch_size=batch_size, is_aug=is_aug, norm=norm)
val_loader = get_loader('validation', val=ds1_val, norm=norm)
mc_loss_fn = MCLoss(type=loss_type)
cross_loss_fn = CrossLoss(type=loss_type)
lpcnn_loss = nn.MSELoss()
A_type = 'Qsm'
tb_writer = SummaryWriter(oj(save_dir, 'tb'))

model = UNet3DAdaINCodeGen().to(device)

from unet_adain_3d_zf import UNetAdaIN3D
model = UNetAdaIN3D(1, 1).to(device)

lambda_grad = 0.05
lambda_tv = 0.01

# from proximal_sti.lib.model.lpcnn.unet3d.models import ResidualUNet3D
# model = ResidualUNet3D(in_channels=1, out_channels=1).to(device)

# model = LPCNN_model(oj(train_loader.dataset.data_root, 'train_gt_mean_partition.npy'),
#                     oj(train_loader.dataset.data_root, 'train_gt_std_partition.npy')).to(device)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=1e-4)

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=1) #0.8)


if lpcnn:
    assert len(train_loader.dataset.sub_wo_cosmos) == 0

print(f"Training {'LPCNN' if lpcnn else 'MOI'} with {'real'if not train_on_sim else 'sim'} data")
print(f"Subjects w. cosmos: {train_loader.dataset.sub_w_cosmos}, subjects w/o cosmos: {train_loader.dataset.sub_wo_cosmos}")
print(f"The validation set is {val_loader.dataset.all_subs}")

# pytorch training loop
best_psnr_diff = -np.inf
total_iter = 0

# mx = 0

for epoch in range(100):
    for i, batch in enumerate(train_loader):
        
        model.train()
        
        batch_phase = batch['phase'].float().to(device)
        batch_mask = batch['mask'].float().to(device)
        batch_cosmos = batch['cosmos'].float().to(device)
        batch_metadata = batch['metadata']
        size_vol = [get_dk_shape(batch_metadata['sub'][i]) for i in range(batch_size)]
        batch_A = []
        affine = batch_metadata['affine'][0]


        # mc_loss, cross_loss, moi_loss, cosmos_loss = torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device
        
        optimizer.zero_grad()
        
        for bn in range(batch_size):
            metadata = {k: v[bn] for k, v in batch_metadata.items()}
            sub, ori = metadata['sub'].item(), metadata['ori'].item()
            data_root = train_loader.dataset.data_root
            H0_path = oj(data_root, 'Sub{0:04d}/ori{1:01d}/Sub{0:04d}_ori{1:01d}.txt'.format(sub, ori))
            Phi = {'H0': np.loadtxt(H0_path)}
            A = construct_A(type=A_type, Phi=Phi, size_vol=size_vol[bn], scaling=scaling_fac).to(device)
            batch_A.append(A)
        
        # x = batch_cosmos
        y = batch_phase.unsqueeze(1)
        
        y = y * batch_mask.unsqueeze(1)
        # xhat = model(y)
        input_adain_code = torch.Tensor([[0.982, 0.982, 1]]).to(device)
        xhat = model(y, input_adain_code)

        # xhat = model(y.unsqueeze(-1), [batch_A[0].dk], batch_mask.unsqueeze(1))#.squeeze(1)

        xhat_den = torch.stack([denormalize(xhat[i].squeeze(0), mean=train_loader.dataset.gt_mean, std=train_loader.dataset.gt_std, mask=batch_mask[i]) for i in range(batch_size)])
        xhat_den_pad = F.pad(xhat_den, (0, size_vol[0][-1] - xhat.shape[-1], 0, size_vol[0][-2] - xhat.shape[-2], 0, size_vol[0][-3] - xhat.shape[-3]))
        Axhat = torch.stack([batch_A[i].forward(xhat_den_pad[i])[:, :xhat[i].shape[1], :xhat[i].shape[2], :xhat[i].shape[3]] for i in range(batch_size)])

        l_cycle = nn.MSELoss()(y[0][batch_mask == 1], Axhat[0][batch_mask == 1]) # ||y-A*xhat||
        # l_cycle = mc_loss_fn(xhat_den, batch_A, y, batch_mask.unsqueeze(1).bool())
        l_grad = gradient_difference_loss(Axhat, y)
        l_tv = 0#total_variation_loss(xhat)

        l_total = l_cycle + lambda_grad * l_grad + lambda_tv * l_tv

        l_total.backward()
        optimizer.step()

        if total_iter % 10 == 0:
            print(f'epoch {epoch}, iter {total_iter}, l_total: {l_total.item()}, l_cycle: {l_cycle.item()}, l_grad: {l_grad.item()}')#, l_tv: {l_tv.item()}')
            # compute psnr
            # x_psnr = x[0].cpu().detach().numpy()
            # xhat_psnr = xhat[0].cpu().detach().numpy()
            # mask_psnr = batch_mask[0].cpu().detach().numpy()

            if batch_metadata['has_cosmos'][0]:
                if norm:
                    xhat = xhat * torch.from_numpy(train_loader.dataset.gt_std)
                    xhat = xhat + torch.from_numpy(train_loader.dataset.gt_mean)

                    x = x * torch.from_numpy(train_loader.dataset.gt_std)
                    x = x + torch.from_numpy(train_loader.dataset.gt_mean)
                psnr_val = qsm_psnr(x[0].cpu().detach().numpy(), xhat[0].cpu().detach().numpy(), 
                                    batch_mask[0].cpu().detach().numpy(), subtract_mean=False)
                print(f'iter {total_iter}, psnr: {psnr_val}')
            
        if total_iter % 800 == 0:
            # save model
            # save images
            print(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            nib.Nifti1Image(xhat.cpu().detach().numpy().squeeze(), affine).to_filename(oj(save_dir, f'iter{total_iter}_xhat.nii.gz'))
            nib.Nifti1Image(batch_mask.cpu().detach().numpy().squeeze(), affine).to_filename(oj(save_dir, f'iter{total_iter}_mask.nii.gz'))
            nib.Nifti1Image(y.cpu().detach().numpy().squeeze(), affine).to_filename(oj(save_dir, f'iter{total_iter}_y.nii.gz'))
            
            # validate
            print('validating at iter', total_iter)
            psnr_vals, ssim_vals = evaluate(model, val_loader, mc_loss_fn, cross_loss_fn, A_type, save_img=False, save_dir=save_dir, 
                                            subset=val_loader.dataset.split, test_on_sim=train_on_sim, scaling_fac=scaling_fac, norm=norm)
            model.train()

            tb_writer.add_scalar('val/psnr moi', np.mean(psnr_vals["moi"]), total_iter)
            tb_writer.add_scalar('val/psnr pinv', np.mean(psnr_vals["pinv"]), total_iter)
            tb_writer.add_scalar('val/ssim moi', np.mean(ssim_vals["moi"]), total_iter)
            tb_writer.add_scalar('val/ssim pinv', np.mean(ssim_vals["pinv"]), total_iter)
            tb_writer.add_scalar('val/psnr moi-pinv', np.mean(psnr_vals["moi"]) - np.mean(psnr_vals["pinv"]), total_iter)
            tb_writer.add_scalar('val/ssim moi-pinv', np.mean(ssim_vals["moi"]) - np.mean(ssim_vals["pinv"]), total_iter)
            # tb_writer.add_scalar('val/lr', scheduler.get_last_lr(), total_iter)
            tb_writer.add_scalar('val/l_cycle', l_cycle, total_iter)
            tb_writer.add_scalar('val/l_grad', l_grad, total_iter)
            tb_writer.add_scalar('val/l_tv', l_tv, total_iter)
            tb_writer.add_scalar('val/l_total', l_total, total_iter)
            if np.mean(psnr_vals["moi"]) - np.mean(psnr_vals["pinv"]) > best_psnr_diff:
                best_psnr_diff = np.mean(psnr_vals["moi"]) - np.mean(psnr_vals["pinv"])
                torch.save(model.state_dict(), oj(save_dir, 'model_best.pth'))

                state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
                torch.save(state, oj(save_dir, 'model_best.pth'))

                print('best model saved')

        total_iter += 1