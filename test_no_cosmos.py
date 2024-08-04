# unsupervised learning with multiple operators

import os

oj = os.path.join
import sys

import nibabel as nib
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch

sys.path.append('.')
sys.path.append('./proximal_sti')
from data import get_loader
from loss import CrossLoss, MCLoss
from lpcnn import LPCNN as LPCNN_model
from model import LPCNN as Model
from proximal_sti.lib.QsmEvaluationToolkit import qsm_psnr, qsm_ssim
from proximal_sti.lib.QsmOperatorToolkit import QsmOperator
from utils import construct_A, sample_A, sample_snr, get_dk_shape


def main():
    device = 'cuda'
    save_dir = 'save_dir/moi_semi101_wobio_vds2_g05'
    scaling_fac = 1.
    load_cosmos = False
    loss_type = 'l2'
    # cross_param = 1
    subset = 'test'
    sep = 'whole'
    norm = True

    print(save_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    # use_pinv = False
    # unet = False
    
    loader = get_loader(subset=subset, sep=sep, norm=norm)
    
    # model_mean = torch.from_numpy(np.load(str(oj(loader.dataset.data_root, 'train_gt_mean_whole.npy')))).float()
    # model_std = torch.from_numpy(np.load(str(oj(loader.dataset.data_root, 'train_gt_std_whole.npy')))).float()
    # model = Model(gt_mean=model_mean, gt_std=model_std, use_pinv=use_pinv, unet=unet, norm=norm).to(device)
    
    # model = LPCNN_model(oj(loader.dataset.data_root, 'train_gt_mean_whole.npy'), 
                        #   oj(loader.dataset.data_root, 'train_gt_std_whole.npy')).to(device)
    
    model = LPCNN_model(oj(loader.dataset.data_root, 'train_gt_mean_partition.npy'), 
                        oj(loader.dataset.data_root, 'train_gt_std_partition.npy')).to(device)


    model.load_state_dict(torch.load(oj(save_dir,'model_best.pth'))['model_state'])
    
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
    for i, batch in enumerate(loader):
        batch_phase = batch['phase'].float().to(device)
        batch_mask = batch['mask'].float().to(device)
        # batch_cosmos = batch['cosmos'].float().to(device)
        batch_metadata = batch['metadata']
        # batch_dk_path = batch['dipole_path']
        size_vol = get_dk_shape(batch_metadata['sub'][0])

        bn = 0
        phase = batch_phase[bn]
        mask = batch_mask[bn]
        # cosmos = batch_cosmos[bn]
        metadata = {k: v[bn] for k, v in batch_metadata.items()}
        affine = metadata['affine']

        # x = cosmos

        y = phase.unsqueeze(0)

        sub, ori = metadata['sub'], metadata['ori']
        data_root = loader.dataset.data_root
        H0_path = oj(data_root, 'Sub{0:04d}/ori{1:01d}/Sub{0:04d}_ori{1:01d}.txt'.format(sub, ori))
        Phi = {'H0': np.loadtxt(H0_path)}
        A = construct_A(type=A_type, Phi=Phi, size_vol=size_vol, scaling=scaling_fac).to(device)
        print(f'saving sub {sub}, the angle is {np.rad2deg(np.linalg.norm(Phi["H0"] - [0, 0, 1]))}')

        xpinv = A.pinv(y, epsilon=1e-2)

        with torch.no_grad():
            # xhat = model(y.unsqueeze(1), [A], mask.unsqueeze(0)).squeeze()
            xhat = model(y.unsqueeze(1).unsqueeze(-1), [A.dk], mask.unsqueeze(0))

        # x = x.cpu().detach().numpy()
        xpinv = xpinv.cpu().detach().numpy()
        xhat = xhat.squeeze()
        xhat = xhat.cpu().detach().numpy()
        # mask = mask.cpu().detach().numpy()

        if norm:
            xhat = xhat * loader.dataset.gt_std
            xhat = xhat + loader.dataset.gt_mean

            # x = x * loader.dataset.gt_std
            # x = x + loader.dataset.gt_mean

            # xpinv = xpinv * loader.dataset.gt_std
            # xpinv = xpinv + loader.dataset.gt_mean

        if save_img:
            nib.Nifti1Image(xhat, affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_xhat.nii.gz'))
            # nib.Nifti1Image(x, np.eye(4)).to_filename(os.path.join(save_dir, f'{subset}{i}_x.nii.gz'))
            if A_type == 'Qsm':
                nib.Nifti1Image(y[0].cpu().detach().numpy(), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_y.nii.gz'))
            else:
                nib.Nifti1Image(y.cpu().detach().numpy(), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_y.nii.gz'))

            nib.Nifti1Image(xpinv, affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_xpinv.nii.gz'))
            nib.Nifti1Image((xhat-xhat.mean()), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_xhat_cen.nii.gz'))
            nib.Nifti1Image((xpinv-xpinv.mean()), affine).to_filename(os.path.join(save_dir, f'{subset}_sub{sub}_ori{ori}_xpinv_cen.nii.gz'))
            # nib.Nifti1Image((x-x.mean()), np.eye(4)).to_filename(os.path.join(save_dir, f'{subset}{i}_x_cen.nii.gz'))


    print(save_dir)



if __name__ == '__main__':
    main()
    