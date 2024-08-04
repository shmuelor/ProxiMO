
import os

oj = os.path.join
import argparse
import sys

import nibabel as nib
import numpy as np
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

sys.path.append('.')
sys.path.append('./proximal_sti')
from data import get_loader
from lpcnn import LPCNN as LPCNN_model
# from model import LPCNN as Model
from proximal_sti.lib.QsmEvaluationToolkit import qsm_psnr, qsm_ssim
# from proximal_sti.lib.QsmOperatorToolkit import QsmOperator
from utils import construct_A, get_dk_shape, sample_A

mean_shift_val = -0.00028766496966350506



def main(save_dir=''):
    device = 'cuda'
    scaling_fac = 1.
    # loss_type = 'l2'
    subset = 'test'
    sep = 'whole'
    norm = True

    print(save_dir)
    
    
    loader = get_loader(subset=subset, test_set=[3,6], sep=sep, norm=norm)
    
    
    model = LPCNN_model(oj(loader.dataset.data_root, f'train_gt_mean_{sep}.npy'), 
                        oj(loader.dataset.data_root, f'train_gt_std_{sep}.npy')).to(device)


    model.load_state_dict(torch.load(oj(save_dir, 'model_best.pth'))['model_state'], strict=False)
    
    A_type = 'Qsm'
    save_img = True
    evaluate(model, loader, A_type, save_img, save_dir, subset=subset, scaling_fac=scaling_fac, norm=norm)


def evaluate(model, loader, A_type, save_img, save_dir, subset, scaling_fac=1., norm=True):
    
    model.eval()
    device = model.parameters().__next__().device
    assert loader.batch_size == 1

    # pytorch testing/validating loop
    psnr_vals = {'proximo': [], 'pinv': []}
    ssim_vals = {'proximo': [], 'pinv': []}
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

        sub, ori = metadata['sub'], metadata['ori']
        data_root = loader.dataset.data_root
        H0_path = oj(data_root, 'Sub{0:04d}/ori{1:01d}/Sub{0:04d}_ori{1:01d}.txt'.format(sub, ori))
        Phi = {'H0': np.loadtxt(H0_path)}
        A = construct_A(type=A_type, Phi=Phi, size_vol=size_vol, scaling=scaling_fac).to(device)

        y_pad = F.pad(y, (0, size_vol[-1] - y.shape[-1], 0, size_vol[-2] - y.shape[-2], 0, size_vol[-3] - y.shape[-3]))
        xpinv = A.pinv(y_pad, epsilon=1e-2)[:y.shape[1], :y.shape[2], :y.shape[3]]

        with torch.no_grad():
            # xhat = model(y.unsqueeze(1), [A], mask.unsqueeze(0)).squeeze()
            xhat = model(y.unsqueeze(1).unsqueeze(-1), [A.dk], mask.unsqueeze(0))

        x = x.cpu().detach().numpy()
        xpinv = xpinv.cpu().detach().numpy()
        xhat = xhat.squeeze()
        xhat = xhat.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()

        if norm:
            xhat = xhat * loader.dataset.gt_std
            xhat = xhat + mean_shift_val

            x = x * loader.dataset.gt_std
            x = x + loader.dataset.gt_mean

        # compute psnr, ssim
        psnr_val = qsm_psnr(x, xhat, mask, subtract_mean=False)
        psnr_val_pinv = qsm_psnr(x, xpinv, mask, subtract_mean=False)
        print(f'testing sub{sub} ori{ori}, psnr: {psnr_val}, psnr_pinv: {psnr_val_pinv}')
        psnr_vals['proximo'].append(psnr_val)
        psnr_vals['pinv'].append(psnr_val_pinv)
        ssim_val = qsm_ssim(x, xhat, subtract_mean=False)
        ssim_val_pinv = qsm_ssim(x, xpinv, subtract_mean=False)
        print(f'testing sub{sub} ori{ori}, ssim: {ssim_val}, ssim_pinv: {ssim_val_pinv}')
        ssim_vals['proximo'].append(ssim_val)
        ssim_vals['pinv'].append(ssim_val_pinv)
        
        if save_img:
            nib.Nifti1Image(xhat, affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_xhat.nii.gz'))
            nib.Nifti1Image(x, affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_x.nii.gz'))
            if A_type == 'Qsm':
                nib.Nifti1Image(y[0].cpu().detach().numpy(), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_y.nii.gz'))
            else:
                nib.Nifti1Image(y.cpu().detach().numpy(), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_y.nii.gz'))

            nib.Nifti1Image(xpinv, affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_xpinv.nii.gz'))
            nib.Nifti1Image((xhat-xhat.mean()), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_xhat_cen.nii.gz'))
            nib.Nifti1Image((xpinv-xpinv.mean()), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_xpinv_cen.nii.gz'))
            nib.Nifti1Image((x-x.mean()), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_x_cen.nii.gz'))


    print(save_dir)
    print(f'psnr (proximo): {np.mean(psnr_vals["proximo"])} +- {np.std(psnr_vals["proximo"])}')
    print(f'psnr (pinv): {np.mean(psnr_vals["pinv"])} +- {np.std(psnr_vals["pinv"])}')
    print(f'ssim (proximo): {np.mean(ssim_vals["proximo"])} +- {np.std(ssim_vals["proximo"])}')
    print(f'ssim (pinv): {np.mean(ssim_vals["pinv"])} +- {np.std(ssim_vals["pinv"])}')
    print('psnr proximo-pinv:', np.mean(psnr_vals["proximo"]) - np.mean(psnr_vals["pinv"]))
    print('ssim proximo-pinv:', np.mean(ssim_vals["proximo"]) - np.mean(ssim_vals["pinv"]))
    print('alpha: %.3f' %(model.alpha.detach().cpu().numpy()))
    
    return psnr_vals, ssim_vals



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test ProxiMO')
    parser.add_argument('-s','--save_dir', help='experiment dir', required=True)
    args = vars(parser.parse_args())
    save_dir = args['save_dir']

    # experiment_path = 'save_dir/exp1'
    main(save_dir=save_dir)
    