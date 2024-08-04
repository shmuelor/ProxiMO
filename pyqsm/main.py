"""Testing code for pyqsm."""
import numpy as np
import nibabel as nib

from tkd import tkd
from utils import save_screenshot

import os
oj = os.path.join

# test tkd
data_dir = '/cis/home/sorenst3/my_documents/unsup_moi/data/'
for sub in os.listdir(data_dir):
    mask_path = oj(data_dir, sub, 'cosmos', f'{sub}_mask.nii.gz')
    if not os.path.exists(mask_path):
        continue
    mask = nib.load(mask_path).get_fdata()

    for ori in os.listdir(oj(data_dir, sub)):
        if not ori.startswith('ori'):
            continue
        
        # load delta
        y_fn = oj(data_dir, sub, ori, f'{sub}_{ori}_phase_norm.nii.gz')
        nii = nib.load(y_fn)
        delta = nii.get_fdata() # / (7 * 42.58)
        voxel_size = nii.header.get_zooms()
        affine = nii.affine

        # load b0dir
        b0_fn = oj(data_dir, sub, ori, f'{sub}_{ori}.txt')
        b0dir = np.loadtxt(b0_fn)
        # b0dir = [-b0dir[0], -b0dir[1], b0dir[2]]

        print(sub, ori, delta.shape, voxel_size, b0dir)

        # run tkd
        chi = tkd(delta, 0.2, b0dir=b0dir, voxel_size=voxel_size)

        # apply brain mask
        # fn = f"{data_dir}/../mask.nii.gz"
        # mask = nib.load(fn).get_fdata()
        chi *= mask

        # save chi
        fn = oj(data_dir, sub, ori, f'{sub}_{ori}_tkd.nii.gz')
        nib.save(nib.Nifti1Image(chi, affine), fn)
        save_screenshot(chi, oj(data_dir, sub, ori, f'{sub}_{ori}_tkd.png'), coor=[45, 108, 103])
        print(f'{sub} {ori} - done')