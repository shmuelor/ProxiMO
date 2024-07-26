import os
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

oj = os.path.join
import torchio as tio
from augmentation import get_transform
from proximal_sti.lib.QsmOperatorToolkit import QsmOperator
from torch.utils.data.sampler import Sampler


class MixStreamsBatchSampler(Sampler):
    """Iterate two sets of indices
    Reference: https://github.com/yulequan/UA-MT/blob/88ed29ad794f877122e542a7fa9505a76fa83515/code/dataloaders/la_heart.py#L162
    """
    def __init__(self, indices1, indices2, batch_size, shuffle=True):
        assert len(indices1) and len(indices2)
        
        self.indices1 = indices1
        self.indices2 = indices2

        self.batch_size = batch_size
        self.shuffle = shuffle


    def __iter__(self):
        iter1 = np.random.permutation(self.indices1) if self.shuffle else self.indices1
        iter2 = np.random.permutation(self.indices2) if self.shuffle else self.indices2
        
        batches = np.array(list(zip(iter1, iter2)))
        if self.shuffle:
            [np.random.shuffle(batch) for batch in batches]

        batches = batches.tolist()
        
        return iter(batches)

    def __len__(self):
        return len(self.indices1) + len(self.indices2)


def get_mixing_sampler(patch_list, batch_size, shuffle=True):

    ind1, ind2 = [], []
    for i, (_, _, _, _, _, has_cosmos) in enumerate(patch_list):
        if has_cosmos:
            ind1.append(i)
        else:
            ind2.append(i)
            
    sampler = MixStreamsBatchSampler(ind1, ind2, batch_size=batch_size, shuffle=shuffle)
    
    return sampler


class MoiDataset(Dataset):
    def __init__(self, sub_w_cosmos=None, sub_wo_cosmos=None, split='train', is_aug=False, sep='partition', k=2, norm=False):
        """
        sub_list: list of subject id
        """
        self.data_root = '/cis/home/sorenst3/my_documents/unsup_moi/data'
        
        self.patch_size = 64
        self.sub_w_cosmos = sub_w_cosmos if sub_w_cosmos is not None else []
        self.sub_wo_cosmos = sub_wo_cosmos if sub_wo_cosmos is not None else []
        self.all_subs = np.concatenate((self.sub_w_cosmos, self.sub_wo_cosmos)).astype(int)
        
        # k is the num of orientations that we use for each subject w/o cosmos. if k >= 4 we can have cosmos for those subjects.
        assert k < 4
        self.k = k
        
        self.split = split
        self.is_aug = is_aug
        
        assert sep == 'partition' or sep == 'whole' 
        self.sep = sep
        
        self.cosmos_oris_dict = self.get_cosmos_oris_dict()
        self.non_cosmos_oris_dict = self.get_non_cosmos_oris_dict()
        
        self.training_transform = get_transform()
        self.patch_list = self.get_patch_list()
        print(self.patch_list)
        self.norm = norm
        if self.norm:

            gt_mean_name = f'train_gt_mean_{sep}.npy'
            gt_std_name = f'train_gt_std_{sep}.npy'

            self.gt_mean = np.load(oj(self.data_root, gt_mean_name))
            self.gt_std = np.load(oj(self.data_root, gt_std_name))

    def __len__(self):
        return len(self.patch_list)
    
    def get_cosmos_oris_dict(self):
        oris_dict = defaultdict(list)
        for sub in self.sub_w_cosmos:
            all_oris = os.listdir(oj(self.data_root, 'Sub{0:04d}'.format(sub)))
            oris = [int(ori[3:]) for ori in all_oris if ori.startswith('ori')]
            oris_dict[sub] = oris
        return oris_dict

    def get_non_cosmos_oris_dict(self):
        oris_dict = defaultdict(list)
        for sub in self.sub_wo_cosmos:
            all_oris = os.listdir(oj(self.data_root, 'Sub{0:04d}'.format(sub)))
            all_oris = [int(ori[3:]) for ori in all_oris if ori.startswith('ori')]   
            oris = np.random.choice(all_oris, size=min(self.k, len(all_oris)), replace=False)
            oris_dict[sub] = oris
        return oris_dict

    def get_patch_list(self):
        l = []
        if self.sep == 'whole':
            for sub in self.all_subs:
                for ori in self.cosmos_oris_dict[sub]:
                    l.append((sub, ori, 0, 0, 0, True))
                for ori in self.non_cosmos_oris_dict[sub]:
                    l.append((sub, ori, 0, 0, 0, False))
        else:
            if self.split == 'train':
                stride = self.patch_size // 4
            else:
                stride = self.patch_size // 2
            for sub in self.sub_w_cosmos:
                for ori in self.cosmos_oris_dict[sub]:
                    l.extend(self.get_patch_details(sub=sub, ori=ori, stride=stride, has_cosmos=True))
            for sub in self.sub_wo_cosmos:
                for ori in self.non_cosmos_oris_dict[sub]:
                    l.extend(self.get_patch_details(sub=sub, ori=ori, stride=stride, has_cosmos=False))
        return l
    
    def get_patch_details(self, sub, ori, stride, has_cosmos):
        l = []
        fn = self.get_fn(sub, ori)
        mask = nib.load(oj(self.data_root, fn['mask'])).get_fdata()
        for x in range(0, mask.shape[0]-self.patch_size, stride):
            for y in range(0, mask.shape[1]-self.patch_size, stride):
                for z in range(0, mask.shape[2]-self.patch_size, stride):
                    if mask[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size].sum() > 0.1 * self.patch_size**3:
                        l.append((sub, ori, x, y, z, has_cosmos))
        return l

    def get_fn(self, sub, ori):
        fn = {
            'img': 'Sub{0:04d}/ori{1:01d}/Sub{0:04d}_ori{1:01d}_phase_norm.nii.gz'.format(sub, ori),
            'mask': 'Sub{0:04d}/cosmos/Sub{0:04d}_mask.nii.gz'.format(sub),
            'cosmos': 'Sub{0:04d}/cosmos/Sub{0:04d}_cosmos.nii.gz'.format(sub),
            'dipole': 'Sub{0:04d}/ori{1:01d}/Sub{0:04d}_ori{1:01d}_dipole.npy'.format(sub, ori)
        }
        return fn
    
    def get_patch(self, idx):
        sub, ori, x, y, z, has_cosmos = self.patch_list[idx]
        fn = self.get_fn(sub, ori)
        phase = nib.load(oj(self.data_root, fn['img'])).get_fdata()
        if self.sep == 'partition':
            phase = phase[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
        
        mask = nib.load(oj(self.data_root, fn['mask'])).get_fdata()
        if self.sep == 'partition':
            mask = mask[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
        
        cosmos = np.zeros_like(mask)
        affine = np.eye(4)
        if os.path.isfile(oj(self.data_root, fn['cosmos'])):
            nib_file = nib.load(oj(self.data_root, fn['cosmos']))
            cosmos = nib_file.get_fdata()
            affine = nib_file.affine
            if self.sep == 'partition':
                cosmos = cosmos[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
            if self.norm:
                cosmos = cosmos - self.gt_mean
                cosmos = cosmos / self.gt_std
                cosmos = cosmos * mask
        
        dipole_path = oj(self.data_root, fn['dipole'])
        
        return phase, mask, cosmos, dipole_path, {'sub': sub, 'ori': ori, 'has_cosmos': has_cosmos, 'affine': affine, 'patch_x': x, 'patch_y': y, 'patch_z': z}

    def aug(self, patch, mask):
        
        # patch = patch[None,:,:,:] # w, h, d -> 1, w, h, d
        # mask = mask[None,:,:,:] # w, h, d -> 1, w, h, d

        sample_tio = tio.Subject(gt=tio.ScalarImage(tensor=patch), brain_mask=tio.LabelMap(tensor=mask))
        sample_tio = self.training_transform(sample_tio)
        patch = sample_tio['gt'].numpy()
        mask = sample_tio['brain_mask'].numpy()[0]

        return patch, mask

    def __getitem__(self, idx):
        phase_patch, mask, cosmos_patch, dipole_path, metadata = self.get_patch(idx)
        if self.split == 'train' and self.is_aug:
            phase_patch, mask = np.expand_dims(phase_patch, axis=0), np.expand_dims(mask, axis=0)
            phase_patch = np.concatenate((phase_patch, np.expand_dims(cosmos_patch, axis=0)))
            mask = np.concatenate((mask, mask))
            phase_patch, mask = self.aug(phase_patch, mask)
            phase_patch, cosmos_patch = phase_patch
        return {'phase': phase_patch, 'mask': mask, 'cosmos': cosmos_patch, 
                'dipole_path': dipole_path + ' ', 'metadata': metadata}

def get_loader(subset='train', train_set_w_cosmos=[], train_set_wo_cosmos=[], val_set=[], test_set=[], batch_size=1, **kwargs):
    # ds1 = [1, 2, 3, 4, 5, 6, 8, 9]
    # ds2 = np.arange(101, 113)
    # bio1 = np.arange(1001, 1101)
    # biocard = np.concatenate((np.arange(1001, 1101), np.arange(2001, 2101), np.arange(3001, 3101),
    #                             np.arange(4001, 4101), np.arange(5001, 5100)))
    # biocard_all = np.concatenate((biocard, np.arange(6001, 6024)))
    
    if subset == 'train':
        dataset = MoiDataset(sub_w_cosmos=train_set_w_cosmos, sub_wo_cosmos=train_set_wo_cosmos,
                             split=subset, small_angles=None, **kwargs)
        if len(train_set_w_cosmos) and len(train_set_wo_cosmos):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=get_mixing_sampler(dataset.patch_list, batch_size=batch_size, shuffle=True),
                num_workers=4,
                pin_memory=True,
            )
        else:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4,
                pin_memory=True,
            )
    elif subset == 'validation':
        dataset = MoiDataset(sub_w_cosmos=val_set, split=subset, small_angles=False, **kwargs)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    elif subset == 'test':
        dataset = MoiDataset(sub_w_cosmos=test_set, split=subset, small_angles=False, **kwargs)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    else:
        raise Exception
    
    return loader
