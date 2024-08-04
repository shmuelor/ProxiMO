# unsupervised learning with multiple operators

import os

oj = os.path.join
import sys

import nibabel as nib
import numpy as np
import torch.nn.functional as F

from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch

sys.path.append('.')
sys.path.append('./proximal_sti')
from data import get_loader
from loss import CrossLoss, MCLoss
from proximal_sti.lib.QsmEvaluationToolkit import qsm_psnr, qsm_ssim
from utils import construct_A, get_dk_shape

from unet3d_adain import UNet3DAdaINCodeGen
from unet_adain_3d_zf import UNetAdaIN3D

import glob

def test_whole_from_partition(save_dir, patch_size=64):
    # loader = get_loader(subset='test', test=np.arange(101, 113), sep='partition', norm=True)
    loader = get_loader(subset='test', test=[3,6], sep='partition', norm=True)
    oris_dict = loader.dataset.get_cosmos_oris_dict()
    psnr_vals = []
    ssim_vals = []
    affine = loader.dataset[0]['metadata']['affine']
    for sub in oris_dict.keys():
        for ori in oris_dict[sub]:
            whole_dim = get_dk_shape(sub)
            mat_sum = np.zeros(whole_dim)
            mat_count = np.zeros(whole_dim)
            patch_names = glob.glob(oj(os.getcwd(), save_dir, f'test_sub{sub}_ori{ori}_px*_py*_pz*_xhat.nii.gz'))
            for patch_name in patch_names:
                locs = patch_name.split('/')[-1].split('_')
                px, py, pz = int(locs[3][2:]), int(locs[4][2:]), int(locs[5][2:])
                patch = nib.load(patch_name).get_fdata()
                mat_sum[px:px + patch_size, py:py + patch_size, pz:pz + patch_size] += patch
                mat_count[px:px + patch_size, py:py + patch_size, pz:pz + patch_size] += 1
            xhat = mat_sum / mat_count
            xhat[np.isnan(xhat)] = 0
            
            x = nib.load(oj(os.getcwd(), 'data', 'Sub{0:04d}'.format(sub), 'cosmos', 'Sub{0:04d}_cosmos.nii.gz'.format(sub))).get_fdata()
            mask = nib.load(oj(os.getcwd(), 'data', 'Sub{0:04d}'.format(sub), 'cosmos', 'Sub{0:04d}_mask.nii.gz'.format(sub))).get_fdata()
            
            nib.Nifti1Image(xhat.squeeze(), affine).to_filename(os.path.join(save_dir, f'test_sub{sub}_ori{ori}_whole_xhat.nii.gz'))

            psnr_val = qsm_psnr(x, xhat, mask, subtract_mean=False)
            psnr_vals.append(psnr_val)
            print(f'testing sub{sub} ori{ori}, psnr: {psnr_val}')
            ssim_val = qsm_ssim(x, xhat, subtract_mean=False)
            ssim_vals.append(ssim_val)
            print(f'testing sub{sub} ori{ori}, ssim: {ssim_val}')

    print(save_dir)
    print(f'psnr (moi): {np.mean(psnr_vals)} +- {np.std(psnr_vals)}')
    print(f'ssim (moi): {np.mean(ssim_vals)} +- {np.std(ssim_vals)}')

save_dir = 'save_dir/unet_adain_noltv_bio_aug_zf'
# save_dir = 'save_dir/unet_adain_noltv_bio_aug_new'
test_whole_from_partition(save_dir)
exit()

def main():
    device = 'cuda'
    save_dir = 'save_dir/unet_adain_noltv_bio_aug_zf'
    scaling_fac = 1.
    load_cosmos = False
    loss_type = 'l2'
    # cross_param = 1
    subset = 'test'
    sep = 'partition'
    norm = True

    print(save_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # loader = get_loader(subset=subset, test=[3,6], sep=sep, norm=norm)
    # loader = get_loader(subset=subset, test=[104], sep=sep, norm=norm)
    loader = get_loader(subset=subset, test=np.arange(101, 113), sep=sep, norm=norm)

    # model = UNet3DAdaINCodeGen().to(device)
    model = UNetAdaIN3D(in_channels=1, out_channels=1).to(device)

    model.load_state_dict(torch.load(oj(save_dir,'model_best.pth'))['model_state'], strict=False)
    
    mc_loss_fn = MCLoss(type=loss_type)
    cross_loss_fn = CrossLoss(type=loss_type)
    A_type = 'Qsm'
    save_img = True
    evaluate(model, loader, mc_loss_fn, cross_loss_fn, A_type, save_img, save_dir, 
             subset=subset, test_on_sim=load_cosmos, scaling_fac=scaling_fac, norm=norm)


def evaluate(model, loader, mc_loss_fn, cross_loss_fn, A_type, save_img, save_dir, subset,
             test_on_sim=False, scaling_fac=3, norm=False):
    
    model.eval()
    device = model.parameters().__next__().device
    assert loader.batch_size == 1

    # pytorch testing/validating loop
    psnr_vals = {'moi': [], 'pinv': []}
    ssim_vals = {'moi': [], 'pinv': []}
    subs_psnrs = defaultdict(list)
    for i, batch in enumerate(loader):
        batch_phase = batch['phase'].float().to(device)
        batch_mask = batch['mask'].float().to(device)
        batch_cosmos = batch['cosmos'].float().to(device)
        batch_metadata = batch['metadata']
        batch_dk_path = batch['dipole_path']
        size_vol = get_dk_shape(batch_metadata['sub'][0])

        mc_loss, cross_loss, cosmos_loss = 0, 0, 0
        
        bn = 0
        phase = batch_phase[bn]
        mask = batch_mask[bn]
        cosmos = batch_cosmos[bn]
        metadata = {k: v[bn] for k, v in batch_metadata.items()}
        affine = metadata['affine']

        x = cosmos

        y = phase.unsqueeze(0)
        # y = y * scaling_fac

        sub, ori = metadata['sub'], metadata['ori']
        data_root = loader.dataset.data_root
        H0_path = oj(data_root, 'Sub{0:04d}/ori{1:01d}/Sub{0:04d}_ori{1:01d}.txt'.format(sub, ori))
        Phi = {'H0': np.loadtxt(H0_path)}
        A = construct_A(type=A_type, Phi=Phi, size_vol=size_vol, scaling=scaling_fac).to(device)

        y_pad = F.pad(y, (0, size_vol[-1] - y.shape[-1], 0, size_vol[-2] - y.shape[-2], 0, size_vol[-3] - y.shape[-3]))
        xpinv = A.pinv(y_pad, epsilon=1e-2)[:y.shape[1], :y.shape[2], :y.shape[3]]

        # for bn in range(batch_size):
        # metadata = {k: v for k, v in batch_metadata.items()}
        # sub, ori = metadata['sub'].item(), metadata['ori'].item()
        # data_root = loader.dataset.data_root
        # H0_path = oj(data_root, 'Sub{0:04d}/ori{1:01d}/Sub{0:04d}_ori{1:01d}.txt'.format(sub, ori))
        # Phi = {'H0': np.loadtxt(H0_path)}
        # A = construct_A(type=A_type, Phi=Phi, size_vol=size_vol, scaling=scaling_fac).to(device)
            # batch_A.append(A)

        with torch.no_grad():
            # xhat = model(y.unsqueeze(0))
            input_adain_code = torch.Tensor([[0.982, 0.982, 1]]).to(device)
            xhat = model(y.unsqueeze(0), input_adain_code)
            # xhat = model(y.unsqueeze(1).unsqueeze(-1), [A.dk], batch_mask.unsqueeze(1))#.squeeze(1)

        x = x.cpu().detach().numpy()
        xpinv = xpinv.cpu().detach().numpy()
        xhat = xhat.squeeze()
        xhat = xhat.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()

        if norm:
            xhat = xhat * loader.dataset.gt_std
            # xhat = xhat + loader.dataset.gt_mean
            xhat = xhat + (-0.00028766496966350506)

            x = x * loader.dataset.gt_std
            x = x + loader.dataset.gt_mean

            # xpinv = xpinv * loader.dataset.gt_std
            # xpinv = xpinv + loader.dataset.gt_mean

        # print(f'testing {i}, total loss: {total_loss.item()}, mc loss: {mc_loss.item()}, cross loss: {cross_loss.item()}')
        # compute psnr, ssim
        psnr_val = qsm_psnr(x, xhat, mask, subtract_mean=False)
        psnr_val_pinv = qsm_psnr(x, xpinv, mask, subtract_mean=False)
        subs_psnrs[int(sub)].append(psnr_val)
        print(f'testing sub{sub} ori{ori}, psnr: {psnr_val}, psnr_pinv: {psnr_val_pinv}')
        psnr_vals['moi'].append(psnr_val)
        psnr_vals['pinv'].append(psnr_val_pinv)
        ssim_val = qsm_ssim(x, xhat, subtract_mean=False)
        ssim_val_pinv = qsm_ssim(x, xpinv, subtract_mean=False)
        print(f'testing sub{sub} ori{ori}, ssim: {ssim_val}, ssim_pinv: {ssim_val_pinv}')
        ssim_vals['moi'].append(ssim_val)
        ssim_vals['pinv'].append(ssim_val_pinv)
        
        if save_img:
            patch_x, patch_y, patch_z = metadata['patch_x'].item(), metadata['patch_y'].item(), metadata['patch_z'].item()
            sep_str = 'whole' if loader.dataset.sep == 'whole' else f'px{patch_x}_py{patch_y}_pz{patch_z}'
            nib.Nifti1Image(xhat.squeeze(), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_{sep_str}_xhat.nii.gz'))
            nib.Nifti1Image(x.squeeze(), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_{sep_str}_x.nii.gz'))
            nib.Nifti1Image(y.cpu().detach().numpy().squeeze(), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_{sep_str}_y.nii.gz'))

            nib.Nifti1Image(xpinv, affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_{sep_str}_xpinv.nii.gz'))
            nib.Nifti1Image((xhat-xhat.mean()), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_{sep_str}_xhat_cen.nii.gz'))
            nib.Nifti1Image((xpinv-xpinv.mean()), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_xpinv_{sep_str}_cen.nii.gz'))
            nib.Nifti1Image((x-x.mean()), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_x_{sep_str}_cen.nii.gz'))


    print(save_dir)
    print(f'psnr (moi): {np.mean(psnr_vals["moi"])} +- {np.std(psnr_vals["moi"])}')
    print(f'psnr (pinv): {np.mean(psnr_vals["pinv"])} +- {np.std(psnr_vals["pinv"])}')
    print(f'ssim (moi): {np.mean(ssim_vals["moi"])} +- {np.std(ssim_vals["moi"])}')
    print(f'ssim (pinv): {np.mean(ssim_vals["pinv"])} +- {np.std(ssim_vals["pinv"])}')
    print('psnr moi-pinv:', np.mean(psnr_vals["moi"]) - np.mean(psnr_vals["pinv"]))
    print('ssim moi-pinv:', np.mean(ssim_vals["moi"]) - np.mean(ssim_vals["pinv"]))
    # print('alpha: %.3f' %(model.alpha.detach().cpu().numpy()))

    print({k: np.mean(v) for k,v in subs_psnrs.items()})

    return psnr_vals, ssim_vals



if __name__ == '__main__':
    main()
    