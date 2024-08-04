# unsupervised learning with multiple operators

import os

oj = os.path.join
import argparse
import sys

import nibabel as nib
import numpy as np
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

print(torch.cuda.current_device())
from tensorboardX import SummaryWriter

sys.path.append('.')
sys.path.append('./proximal_sti')
from test import evaluate

import torch.nn as nn
import torch.optim as optim
from data import get_loader
from loss import CrossLoss, MCLoss
from lpcnn import LPCNN as LPCNN_model
from proximal_sti.lib.QsmEvaluationToolkit import qsm_psnr
from utils import construct_A, denormalize, get_dk_shape, sample_A, sample_snr

is_aug = False 
scaling_fac = 1.
loss_type = 'l2'
batch_size = 2
test_sep = 'whole'
norm = True

x_avg_max = 0.6
y_avg_max = 8.5e-5

beta = 0.5
gamma = 0.8

A_type = 'Qsm'

device = 'cuda'

ds1_train = [1,2,4,5,8,9]
ds1_val = [3,6]
# ds2 = np.arange(101, 113)
# biocard = np.concatenate((np.arange(1001, 1101), np.arange(2001, 2101), np.arange(3001, 3101),
#                             np.arange(4001, 4101), np.arange(5001, 5100)))
# biocard_all = np.concatenate((biocard, np.arange(6001, 6024)))

def main(unsupervised, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    mc_loss_fn = MCLoss(type=loss_type)
    cross_loss_fn = CrossLoss(type=loss_type)
    lpcnn_loss = nn.MSELoss()

    tb_writer = SummaryWriter(oj(save_dir, 'tb'))

    val_loader = get_loader('validation', val_set=ds1_val, sep=test_sep, norm=norm)
    if unsupervised:
        train_loader = get_loader('train', train_set_w_cosmos=[], train_set_wo_cosmos=ds1_train, batch_size=batch_size, is_aug=is_aug, norm=norm)
    else:
        train_loader = get_loader('train', train_set_w_cosmos=[2,5,9], train_set_wo_cosmos=ds1_train, batch_size=batch_size, is_aug=is_aug, norm=norm)
    
    model = LPCNN_model(oj(train_loader.dataset.data_root, 'train_gt_mean_partition.npy'),
                    oj(train_loader.dataset.data_root, 'train_gt_std_partition.npy')).to(device)
    
    model_mean = train_loader.dataset.gt_mean if norm else 0
    model_std = train_loader.dataset.gt_std if norm else 1

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) if unsupervised else torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=1)

# if lpcnn:
    # assert len(train_loader.dataset.sub_wo_cosmos) == 0

    print(f"Training 'ProxiMO' {'unsupervised' if unsupervised else 'semi-supervised'}...")
    print(f"Subjects w. cosmos: {train_loader.dataset.sub_w_cosmos}, subjects w/o cosmos: {train_loader.dataset.sub_wo_cosmos}")
    print(f"The validation set is {val_loader.dataset.all_subs}")

    # pytorch training loop
    best_psnr_diff = -np.inf
    total_iter = 0

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
            affine = batch_metadata['affine'][0]

            mc_loss, cross_loss, proximo_loss, cosmos_loss = torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device)
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
            y = batch_phase.unsqueeze(1)
            y = y * batch_mask.unsqueeze(1)
            
            # xhat = model(y, batch_A, batch_mask).squeeze()
            xhat = model(y.unsqueeze(-1), [batch_A[0].dk, batch_A[1].dk], batch_mask.unsqueeze(1)).squeeze(1)

            cos_loss = lpcnn_loss(xhat[batch_metadata['has_cosmos']], x[batch_metadata['has_cosmos']])
                    
            cosmos_loss += cos_loss / x_avg_max if not torch.isnan(cos_loss) else torch.tensor(0)

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
            cross = cross_loss_fn(xhat_cross, xhat)  # cross operator loss
            cross_loss += cross / x_avg_max

            proximo_loss = beta * mc_loss + (1 - beta) * cross_loss

            total_loss = gamma * cosmos_loss + (1 - gamma) * proximo_loss

            scheduler.step(epoch)

            total_loss.backward()
            optimizer.step()

            if total_iter % 10 == 0:
                print(f'epoch {epoch}, iter {total_iter}, cosmos loss: {cosmos_loss.item()}, mc loss: {mc_loss.item()}, cross loss: {cross_loss.item()}, total loss: {total_loss.item()}')
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
                nib.Nifti1Image(xhat.cpu().detach().numpy(), affine).to_filename(oj(save_dir, f'iter{total_iter}_xhat.nii.gz'))
                nib.Nifti1Image(batch_mask.cpu().detach().numpy(), affine).to_filename(oj(save_dir, f'iter{total_iter}_mask.nii.gz'))
                if A_type == 'Qsm':
                    nib.Nifti1Image(y[0].cpu().detach().numpy(), affine).to_filename(oj(save_dir, f'iter{total_iter}_y.nii.gz'))
                else:
                    nib.Nifti1Image(y.cpu().detach().numpy(), affine).to_filename(oj(save_dir, f'iter{total_iter}_y.nii.gz'))
                
                # validate
                print('validating at iter', total_iter)
                psnr_vals, ssim_vals = evaluate(model, val_loader, A_type, save_img=False, save_dir=save_dir, 
                                                subset=val_loader.dataset.split, scaling_fac=scaling_fac, norm=norm)
                model.train()

                tb_writer.add_scalar('val/psnr proximo', np.mean(psnr_vals["proximo"]), total_iter)
                tb_writer.add_scalar('val/psnr pinv', np.mean(psnr_vals["pinv"]), total_iter)
                tb_writer.add_scalar('val/ssim proximo', np.mean(ssim_vals["proximo"]), total_iter)
                tb_writer.add_scalar('val/ssim pinv', np.mean(ssim_vals["pinv"]), total_iter)
                tb_writer.add_scalar('val/psnr proximo-pinv', np.mean(psnr_vals["proximo"]) - np.mean(psnr_vals["pinv"]), total_iter)
                tb_writer.add_scalar('val/ssim proximo-pinv', np.mean(ssim_vals["proximo"]) - np.mean(ssim_vals["pinv"]), total_iter)
                tb_writer.add_scalar('val/lr', scheduler.get_last_lr(), total_iter)
                tb_writer.add_scalar('val/cosmos_loss', cosmos_loss, total_iter)
                tb_writer.add_scalar('val/mc_loss', mc_loss, total_iter)
                tb_writer.add_scalar('val/cross_loss', cross_loss, total_iter)
                tb_writer.add_scalar('val/total_loss', total_loss, total_iter)
                if np.mean(psnr_vals["proximo"]) - np.mean(psnr_vals["pinv"]) > best_psnr_diff:
                    best_psnr_diff = np.mean(psnr_vals["proximo"]) - np.mean(psnr_vals["pinv"])
                    torch.save(model.state_dict(), oj(save_dir, 'model_best.pth'))

                    state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
                    torch.save(state, oj(save_dir, 'model_best.pth'))

                    print('best model saved')

            total_iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run ProxiMO')
    parser.add_argument('-s','--save_dir', help='dir to save results to', required=True)
    parser.add_argument('-u','--unsupervised', help='True if unsupervised experiment, else False', required=True)
    args = vars(parser.parse_args())
    unsupervised = args['unsupervised']
    save_dir = args['save_dir']
    
    # save_dir = 'save_dir/exp1'
    # unsupervised = True
    main(unsupervised=unsupervised, save_dir=save_dir)
