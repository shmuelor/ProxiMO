# unsupervised learning with multiple operators

import os

oj = os.path.join
import sys

import nibabel as nib
import numpy as np
import scipy.io
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch

# print(torch.cuda.device_count())
print(torch.cuda.current_device())
from tensorboardX import SummaryWriter

sys.path.append('.')
sys.path.append('./proximal_sti')
from test import evaluate

import torch.nn as nn
import torch.optim as optim
from data import get_loader
# from pdata import get_loader
from loss import CrossLoss, MCLoss
from lpcnn import LPCNN as LPCNN_model
from lpcnn import _ifft, _rfft
from model import LPCNN as Model
from proximal_sti.lib.QsmEvaluationToolkit import qsm_psnr, qsm_ssim
from proximal_sti.lib.QsmOperatorToolkit import QsmOperator
from utils import (construct_A, denormalize, get_dk_shape, normalize, sample_A,
                   sample_snr)

lpcnn = False
train_on_sim = False
is_aug = False 
scaling_fac = 1.
loss_type = 'l2'
lr = 1e-4
batch_size = 2
test_sep = 'whole'
norm = True
unet = False
# mc_fac = 5e3

x_avg_max = 0.6
y_avg_max = 8.5e-5
beta = 0.5
gamma = 0.8

device = 'cuda'
save_dir = 'save_dir/moi_semi_ds2_wobio_test_ang5_g08'
os.makedirs(save_dir, exist_ok=True)
use_pinv = False

# model.load_state_dict(torch.load(oj(save_dir,'model_best.pth')))

train_loader = get_loader('train', batch_size=batch_size, is_aug=is_aug, norm=norm)
test_loader = get_loader('test', sep=test_sep, norm=norm)
mc_loss_fn = MCLoss(type=loss_type)
cross_loss_fn = CrossLoss(type=loss_type)
lpcnn_loss = nn.MSELoss()
A_type = 'Qsm'
tb_writer = SummaryWriter(oj(save_dir, 'tb'))

model_mean = train_loader.dataset.gt_mean if norm else 0
model_std = train_loader.dataset.gt_std if norm else 1
# model = Model(gt_mean=model_mean, gt_std=model_std,
#               unet=unet, use_pinv=use_pinv, norm=norm).to(device)

model = LPCNN_model(oj(train_loader.dataset.data_root, 'train_gt_mean_partition.npy'),
                    oj(train_loader.dataset.data_root, 'train_gt_std_partition.npy')).to(device)

# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=1) #0.8)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)


if lpcnn:
    assert len(train_loader.dataset.sub_wo_cosmos) == 0

print(f"Training {'LPCNN' if lpcnn else 'MOI'} with {'real'if not train_on_sim else 'sim'} data")
print(f"Subjects w. cosmos: {train_loader.dataset.sub_w_cosmos}, subjects w/o cosmos: {train_loader.dataset.sub_wo_cosmos}")
print(f"The test set is {test_loader.dataset.all_subs}")

# pytorch training loop
best_psnr_diff = -np.inf
total_iter = 0

mx = 0

for epoch in range(800):
    for i, batch in enumerate(train_loader):
        
        model.train()
        
        batch_phase = batch['phase'].float().to(device)
        batch_mask = batch['mask'].float().to(device)
        batch_cosmos = batch['cosmos'].float().to(device)
        batch_metadata = batch['metadata']
        batch_dk_path = batch['dipole_path']
        size_vol = [get_dk_shape(batch_metadata['sub'][i]) for i in range(batch_size)]
        batch_A = []
        # print(batch_metadata['has_cosmos'])

        mc_loss, cross_loss, moi_loss, cosmos_loss = torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device)
        for bn in range(batch_size):
            metadata = {k: v[bn] for k, v in batch_metadata.items()}
            sub, ori = metadata['sub'].item(), metadata['ori'].item()
            data_root = train_loader.dataset.data_root
            H0_path = oj(data_root, 'Sub{0:04d}/ori{1:01d}/Sub{0:04d}_ori{1:01d}.txt'.format(sub, ori))
            Phi = {'H0': np.loadtxt(H0_path)}
            A = construct_A(type=A_type, Phi=Phi, size_vol=size_vol[bn], scaling=scaling_fac).to(device)
            batch_A.append(A)
        
        optimizer.zero_grad()
        
        x = batch_cosmos
        if train_on_sim:
            x_den = torch.stack([denormalize(x[i], mean=train_loader.dataset.gt_mean, std=train_loader.dataset.gt_std, mask=batch_mask[i]) for i in range(batch_size)])
            x_den_pad = F.pad(x_den, (0, size_vol[-1]-x.shape[-1], 0, size_vol[-2]-x.shape[-2], 0, size_vol[-3]-x.shape[-3]))
            y = [batch_A[i].forward(x_den_pad[i])[:, :x.shape[1], :x.shape[2], :x.shape[3]] for i in range(batch_size)]
            # y = [] # TODO - add noise?
            y = torch.stack(y)
        else:
            y = batch_phase.unsqueeze(1)
        
        y = y * batch_mask.unsqueeze(1)
        # xhat = model(y, batch_A, batch_mask).squeeze()
        xhat = model(y.unsqueeze(-1), [batch_A[0].dk, batch_A[1].dk], batch_mask.unsqueeze(1)).squeeze()

        cos_loss = lpcnn_loss(xhat[batch_metadata['has_cosmos']], x[batch_metadata['has_cosmos']])
                
        cosmos_loss += cos_loss / x_avg_max if not torch.isnan(cos_loss) else torch.tensor(0)

        if not lpcnn:
            xhat_den = torch.stack([denormalize(xhat[i], mean=train_loader.dataset.gt_mean, std=train_loader.dataset.gt_std, mask=batch_mask[i]) for i in range(batch_size)])
            mc = mc_loss_fn(xhat_den, batch_A, y, batch_mask.unsqueeze(1).bool())  # measurement consistency loss
            mc_loss += mc / y_avg_max

            y_cross = torch.zeros_like(y)
            A_cross = []
            for bn in range(batch_size):
                A_cross_bn = sample_A(type=A_type, size_vol=size_vol[bn], scaling=scaling_fac).to(device)
                _, dk_dim0, dk_dim1, dk_dim2 = A_cross_bn.dk.shape
                _, x_dim0, x_dim1, x_dim2 = xhat_den.shape
                xhat_den_pad = F.pad(xhat_den[bn], (0, dk_dim2-x_dim2, 0, dk_dim1-x_dim1, 0, dk_dim0-x_dim0))
                y_cross_bn = A_cross_bn.forward(xhat_den_pad)[:, :x_dim0, :x_dim1, :x_dim2]
                A_cross_bn.add_noise(y_cross_bn, snr=sample_snr(10, 20))
                y_cross[bn] = y_cross_bn
                A_cross.append(A_cross_bn)

            y_cross = y_cross * batch_mask.unsqueeze(1)

            # xhat_cross = model(y_cross, A_cross, batch_mask).squeeze()
            xhat_cross = model(y_cross.unsqueeze(-1), [A_cross[0].dk, A_cross[1].dk], batch_mask.unsqueeze(1)).squeeze()
            cross = cross_loss_fn(xhat_cross, xhat) # cross operator loss
            cross_loss += cross / x_avg_max

            # if total_iter > 5000:
            #     beta = 0.75
            
            moi_loss = beta * mc_loss + (1 - beta) * cross_loss
            # moi_loss = 5e3 * mc_loss + cross_loss

        # mc_fac = 1e4 if cosmos_loss == 0 else 5e3
        # total_loss = cosmos_loss + moi_loss
        total_loss = gamma * cosmos_loss + (1 - gamma) * moi_loss

        scheduler.step(epoch)

        total_loss.backward()
        optimizer.step()

        if total_iter % 10 == 0:
            print(f'epoch {epoch}, iter {total_iter}, cosmos loss: {cosmos_loss.item()}, mc loss: {mc_loss.item()}, cross loss: {cross_loss.item()}, total loss: {total_loss.item()}')
            # compute psnr
            x_psnr = x[0].cpu().detach().numpy()
            xhat_psnr = xhat[0].cpu().detach().numpy()
            mask_psnr = batch_mask[0].cpu().detach().numpy()

            if batch_metadata['has_cosmos'][0]:
                if norm:
                    xhat_psnr = xhat_psnr * train_loader.dataset.gt_std
                    xhat_psnr = xhat_psnr + train_loader.dataset.gt_mean

                    x_psnr = x_psnr * train_loader.dataset.gt_std
                    x_psnr = x_psnr + train_loader.dataset.gt_mean
                psnr_val = qsm_psnr(x_psnr, xhat_psnr, mask_psnr, subtract_mean=False)
                print(f'iter {total_iter}, psnr: {psnr_val}')
            
        if total_iter % 800 == 0:
            # save model
            # save images
            print(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            nib.Nifti1Image(xhat.cpu().detach().numpy(), np.eye(4)).to_filename(oj(save_dir, f'iter{total_iter}_xhat.nii.gz'))
            nib.Nifti1Image(batch_mask.cpu().detach().numpy(), np.eye(4)).to_filename(oj(save_dir, f'iter{total_iter}_mask.nii.gz'))
            if lpcnn: # has_cosmos
                nib.Nifti1Image(x.cpu().detach().numpy(), np.eye(4)).to_filename(oj(save_dir, f'iter{total_iter}_x.nii.gz'))
            if A_type == 'Qsm':
                nib.Nifti1Image(y[0].cpu().detach().numpy(), np.eye(4)).to_filename(oj(save_dir, f'iter{total_iter}_y.nii.gz'))
            else:
                nib.Nifti1Image(y.cpu().detach().numpy(), np.eye(4)).to_filename(oj(save_dir, f'iter{total_iter}_y.nii.gz'))
            # test
            print('validating at iter', total_iter)
            psnr_vals, ssim_vals = evaluate(model, test_loader, mc_loss_fn, cross_loss_fn, A_type, save_img=False, save_dir=save_dir, 
                                            subset=test_loader.dataset.split, test_on_sim=train_on_sim, scaling_fac=scaling_fac, norm=norm)
            model.train()

            tb_writer.add_scalar('test/psnr moi', np.mean(psnr_vals["moi"]), total_iter)
            tb_writer.add_scalar('test/psnr pinv', np.mean(psnr_vals["pinv"]), total_iter)
            tb_writer.add_scalar('test/ssim moi', np.mean(ssim_vals["moi"]), total_iter)
            tb_writer.add_scalar('test/ssim pinv', np.mean(ssim_vals["pinv"]), total_iter)
            tb_writer.add_scalar('test/psnr moi-pinv', np.mean(psnr_vals["moi"]) - np.mean(psnr_vals["pinv"]), total_iter)
            tb_writer.add_scalar('test/ssim moi-pinv', np.mean(ssim_vals["moi"]) - np.mean(ssim_vals["pinv"]), total_iter)
            tb_writer.add_scalar('test/lr', scheduler.get_last_lr(), total_iter)
            tb_writer.add_scalar('test/cosmos_loss', cosmos_loss, total_iter)
            tb_writer.add_scalar('test/mc_loss', mc_loss, total_iter)
            tb_writer.add_scalar('test/cross_loss', cross_loss, total_iter)
            tb_writer.add_scalar('test/total_loss', total_loss, total_iter)
            if np.mean(psnr_vals["moi"]) - np.mean(psnr_vals["pinv"]) > best_psnr_diff:
                best_psnr_diff = np.mean(psnr_vals["moi"]) - np.mean(psnr_vals["pinv"])
                # torch.save(model.state_dict(), oj(save_dir, 'model_best.pth'))

                state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
                torch.save(state, oj(save_dir, 'model_best.pth'))

                print('best model saved')

        total_iter += 1