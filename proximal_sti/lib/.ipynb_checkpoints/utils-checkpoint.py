import os
import numpy as np
from tqdm import tqdm
import yaml
try:
    from skimage.measure import compare_ssim as ssim
    from skimage.measure import compare_psnr as psnr
except:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data.sampler import Sampler
from torch.utils.data import WeightedRandomSampler
from lib.dataset.sti_dataset import STIDataset
from lib.dataset.ext_dataset import ExtDataset
from lib.dataset.sti_dataset_efficient import STIDatasetEfficient

from lib.loss.ssim import ssim3d

#from lib.optimizer.sgdadam.sgdadam import sgdadam
from lib.model.lpcnn.lpcnn import LPCNN
from lib.model.lpcnn.lpcnn_unet import LPCNN as LPCNN_UNET
from lib.model.lpcnn.lpcnn_resunet import LPCNN as LPCNN_RESUNET
from lib.model.lpcnn_indep.lpcnn_indep import LPCNN_INDEP
from lib.model.vdsrr.vdsrr import VDSRR
from lib.model.neumann.neumann import NEUMANN

def getSampler(args, root_dir, sep='partition', split='train'):
    dataset_path = root_dir
    
    if sep == 'partition':
        root_path = dataset_path + sep + '/partition_data{}_list/'.format(args.number)
    elif sep == 'whole':
        root_path = dataset_path + sep + '/data{}_list/'.format(args.number)
        
    input_data = []
    if split == 'train':
        input_list_file = os.path.join(root_path, args.train_list)
    with open(input_list_file, 'r') as f:
        for line in f:
            input_data.append(line.rstrip('\n'))
    
    group1 = ['Sub001', 'Sub002', 'Sub003']
    group2 = ['Sub005', 'Sub006', 'Sub007', 'Sub008', 'Sub009']
    ind1, ind2 = [], []
    for i, data in enumerate(input_data):
        SubName = data.split(' ')[0]
        if SubName in group1:
            ind1.append(i)
        elif SubName in group2:
            ind2.append(i)
        else:
            raise
            
    sampler = TwoStreamBatchSampler(ind1, ind2, args.batch_size)
    
    return sampler
    
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    Reference: https://github.com/yulequan/UA-MT/blob/88ed29ad794f877122e542a7fa9505a76fa83515/code/dataloaders/la_heart.py#L162
    """
    def __init__(self, primary_indices, secondary_indices, batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.batch_size = batch_size

        assert len(self.primary_indices) >= self.batch_size > 0
        assert len(self.secondary_indices) >= self.batch_size > 0

    def __iter__(self):
        primary_iter = np.random.permutation(self.primary_indices)
        secondary_iter = np.random.permutation(self.secondary_indices)
        
        def grouper(iterable, n):
            "Collect data into fixed-length chunks or blocks"
            # grouper('ABCDEFG', 3) --> ABC DEF"
            args = [iter(iterable)] * n
            return zip(*args)
        
        primary_batches = list(grouper(primary_iter, self.batch_size))
        secondary_batches = list(grouper(secondary_iter, self.batch_size))

        batches = np.random.permutation(primary_batches + secondary_batches).tolist()
        print(batches[:10])
        
        return iter(batches)

    def __len__(self):
        return len(self.primary_indices) // self.batch_size + len(self.secondary_indices) // self.batch_size


def getWeightedSampler(args, root_dir, sep='partition', split='train'):
    dataset_path = root_dir
    
    if sep == 'partition':
        root_path = dataset_path + sep + '/partition_data{}_list/'.format(args.number)
    elif sep == 'whole':
        root_path = dataset_path + sep + '/data{}_list/'.format(args.number)
        
    input_data = []
    if split == 'train':
        input_list_file = os.path.join(root_path, args.train_list)
    with open(input_list_file, 'r') as f:
        for line in f:
            input_data.append(line.rstrip('\n'))
    
    group1 = ['Sub001', 'Sub002', 'Sub003']
    group2 = ['Sub005', 'Sub006', 'Sub007', 'Sub008', 'Sub009']
    ind1, ind2 = [], []
    for i, data in enumerate(input_data):
        SubName = data.split(' ')[0]
        if SubName in group1:
            ind1.append(i)
        elif SubName in group2:
            ind2.append(i)
        else:
            raise
    
    # get weights
    i, c =np.unique([x.split(' ')[0] for x in input_data], return_counts=True)
    c = c/sum(c)
    w = []
    for x in ind1 + ind2:
        w.append(1 / c[np.argwhere(i == input_data[x].split(' ')[0])[0,0]])
    
        
    sampler = WeightedTwoStreamBatchSampler(ind1, ind2, w, args.batch_size, int(args.samples_per_epoch / args.batch_size +10))
    
    return sampler

    
class WeightedTwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices with weights
    Reference: https://github.com/yulequan/UA-MT/blob/88ed29ad794f877122e542a7fa9505a76fa83515/code/dataloaders/la_heart.py#L162
    """
    def __init__(self, primary_indices, secondary_indices, weights, batch_size, num_batch):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.weights = weights
        self.batch_size = batch_size
        self.num_batch = num_batch

        assert len(self.primary_indices) >= self.batch_size > 0
        assert len(self.secondary_indices) >= self.batch_size > 0

    def __iter__(self):
        inds = list(WeightedRandomSampler(self.weights, self.num_batch * self.batch_size, replacement=False))
        primary_indices = [x for x in inds if x in self.primary_indices]
        secondary_indices = [x for x in inds if x in self.secondary_indices]
        
        primary_iter = np.random.permutation(primary_indices)
        secondary_iter = np.random.permutation(secondary_indices)
        
        def grouper(iterable, n):
            "Collect data into fixed-length chunks or blocks"
            # grouper('ABCDEFG', 3) --> ABC DEF"
            args = [iter(iterable)] * n
            return zip(*args)
        
        primary_batches = list(grouper(primary_iter, self.batch_size))
        secondary_batches = list(grouper(secondary_iter, self.batch_size))

        batches = np.random.permutation(primary_batches + secondary_batches).tolist()
        print(batches[:10])
        self.batches = batches
        
        return iter(batches)
    
    def __len__(self):
        return self.num_batch

def prepareDataset(args, device, root_dir, data_aug=None, normalize=False, n_w=16):
    
    if args.dataset.lower() == 'sti':

        dataset_path = root_dir #+ 'sti_dataset/sti_sub/synthetic20_data/'

        train_dataset = STIDataset(args, dataset_path, device, split='train', tesla=args.tesla, number=args.number, snr=args.snr, is_norm=normalize, patch_size=args.patch_size, dk_size=args.dk_size)
        val_dataset = STIDataset(args, dataset_path, device, split='validate', sep='whole', tesla=args.tesla, number=args.number, snr=args.snr, is_norm=normalize, patch_size=args.patch_size, dk_size=args.dk_size)
        test_dataset = STIDataset(args, dataset_path, device, split='test', sep='whole', tesla=args.tesla, number=args.number, snr=args.snr, is_norm=normalize, patch_size=args.patch_size, dk_size=args.dk_size)
    
    elif args.dataset.lower() == 'stieff':

        dataset_path = root_dir

        train_dataset = STIDatasetEfficient(args, dataset_path, device, split='train', tesla=args.tesla, number=args.number, snr=args.snr, is_norm=normalize, patch_size=args.patch_size, dk_size=args.dk_size)
        val_dataset = STIDatasetEfficient(args, dataset_path, device, split='validate', sep='whole', tesla=args.tesla, number=args.number, snr=args.snr, is_norm=normalize, patch_size=args.patch_size, dk_size=args.dk_size)
        test_dataset = STIDatasetEfficient(args, dataset_path, device, split='test', sep='whole', tesla=args.tesla, number=args.number, snr=args.snr, is_norm=normalize, patch_size=args.patch_size, dk_size=args.dk_size)
    
    else:
        raise ValueError('unknown dataset: ' + dataset)

    if args.use_sampler > 0:
        if args.use_sampler == 1:
            sampler = getSampler(args, root_dir, sep='partition', split='train')
        elif args.use_sampler == 2:
            sampler = getWeightedSampler(args, root_dir, sep='partition', split='train')
        try:
            train_loader = data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=n_w, pin_memory=False, prefetch_factor=1)
            val_loader = data.DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=False, prefetch_factor=1)
            test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=False, prefetch_factor=1)
            print('++++++', n_w)
        except:
            train_loader = data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=n_w, pin_memory=False)
            val_loader = data.DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=False)
            test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=False)
            print('------')
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=n_w, shuffle=True, drop_last=True, pin_memory=False)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, pin_memory=False)

    print('Got {} training examples'.format(len(train_loader.dataset)))
    print('Got {} validation examples'.format(len(val_loader.dataset)))
    print('Got {} test examples'.format(len(test_loader.dataset)))
    
    return train_loader, val_loader, test_loader

def loadData(args, device, root_dir, prediction_data, normalize=False):

    if args.dataset == 'sti':

        prediction_set, subject_num, ori_num, patch_num, case = prediction_data

        if case == 'whole':
            dataset_path = root_dir #+ 'sti_dataset/sti_sub/synthetic20_data/'
            separation = 'whole'
        else:
            dataset_path = root_dir #+ 'sti_dataset/sti_sub/synthetic20_data/'
            separation = 'partition'

        if prediction_set == 'train':
            ext_data = STIDataset(args, dataset_path, device, split='train', sep=separation, tesla=args.tesla, number=args.number, snr=args.snr, is_norm=normalize)
        elif prediction_set == 'val':
            ext_data = STIDataset(args, dataset_path, device, split='validate', sep=separation, tesla=args.tesla, number=args.number, snr=args.snr, is_norm=normalize)
        elif prediction_set == 'test':
            ext_data = STIDataset(args, dataset_path, device, split='test', sep=separation, tesla=args.tesla, number=args.number, snr=args.snr, is_norm=normalize)
        elif prediction_set == 'ext':
            with open(args.ext_data, 'r') as f:
                data_info = yaml.safe_load(f)
            ext_data = ExtDataset(data_info, device, normalize)
        else:
            raise ValueError('Unknown extra data category: ' + prediction_set)

    else:
        raise ValueError('unknown dataset: ' + args.dataset)
        
    data_loader = data.DataLoader(ext_data, batch_size=1, num_workers=1, shuffle=False)
    
    print('Got {} testing examples'.format(len(data_loader.dataset)))

    return data_loader
'''
def ext_handle(args, root_dir, prediction_data, normalize, dataset_path):

    prediction_set, subject_num, ori_num, patch_num, case = prediction_data

    input_name = 'ext_phase.txt'
    gt_name = 'ext_gt.txt'
    mask_name = 'ext_mask.txt'
    ang_name = 'ext_ang.txt'
    #erod_mask_name = 'ext_erod_mask.txt'
    
    temp_sub = subject_num + '/' + ori_num

    if case == 'whole':
        with open(dataset_path + input_name, 'w') as f:
            f.write(root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/phase_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_LBVSMV.npy\n')
        with open(dataset_path + gt_name, 'w') as f:
            f.write(root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/cosmos_data/' + subject_num + '/' + subject_num + '_cosmos.npy\n')
        with open(dataset_path + mask_name, 'w') as f:
            f.write(root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/mask_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_mask.npy\n')
        with open(dataset_path + ang_name, 'w') as f:
            f.write(root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/angle_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_ang.npy\n')
        #with open(dataset_path + erod_mask_name, 'w') as f:
        #   f.write(root_dir + 'hcp_dataset/dti_dataset/whole/erod_mask_data/' + temp_sub + '_erod_mask.npy\n')


    elif case == 'patch':
        with open(dataset_path + input_name, 'w') as f:
            f.write(root_dir + 'qsm_dataset/qsm_B_r/mix_data/partition/phase_pdata' + temp_sub + '/' + subject_num + '_' + ori_num + '_LBVSMV_p' + patch_num + '.npy\n')
        with open(dataset_path + gt_name, 'w') as f:
            f.write(root_dir + 'qsm_dataset/qsm_B_r/mix_data/partition/cosmos_pdata/' + temp_sub + '/' + subject_num + '_' + ori_num + '_cosmos_p' + patch_num + '.npy\n')
        with open(dataset_path + mask_name, 'w') as f:
            f.write(root_dir + 'qsm_dataset/qsm_B_r/mix_data/partition/mask_pdata/' + temp_sub + '/' + subject_num + '_' + ori_num + '_mask_p' + patch_num + '.npy\n')
        with open(dataset_path + ang_name, 'w') as f:
            f.write(root_dir + 'qsm_dataset/qsm_B_r/mix_data/whole/angle_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_ang.npy\n')
        #with open(dataset_path + erod_mask_name, 'w') as f:
        #   f.write(root_dir + 'hcp_dataset/dti_dataset/partition/erod_mask_pdata/' + temp_sub + '_erod_mask_p' + patch_num + '.npy\n')
            
    else:
        raise ValueError('unknown case: ' + case)

    ext_data = QsmDataset(dataset_path, split='ext', tesla=args.tesla, is_norm=normalize)

    return ext_data
'''

def chooseModel(args, root_dir):
    
    model = None
    if args.model_arch.lower() == 'lpcnn':
        model = LPCNN(root_dir + 'partition/partition_data6_list/train_gt_mean.npy', root_dir + 'partition/partition_data6_list/train_gt_std.npy', args.iter_num, args.feat_dim, args.num_blocks)
    elif args.model_arch.lower() == 'lpcnn_unet':
        model = LPCNN_UNET(root_dir + 'partition/partition_data6_list/train_gt_mean.npy', root_dir + 'partition/partition_data6_list/train_gt_std.npy', args.iter_num, args.feat_dim, args.num_blocks)
    elif args.model_arch.lower() == 'lpcnn_resunet':
        model = LPCNN_RESUNET(root_dir + 'partition/partition_data6_list/train_gt_mean.npy', root_dir + 'partition/partition_data6_list/train_gt_std.npy', args.iter_num, args.feat_dim, args.num_blocks, args.train_step_size)
    elif args.model_arch.lower() == 'lpcnn_indep':
        model = LPCNN_INDEP(root_dir + 'partition/partition_data6_list/train_gt_mean.npy', root_dir + 'partition/partition_data6_list/train_gt_std.npy')
    elif args.model_arch.lower() == 'vdsrr':
        model = VDSRR(root_dir + 'partition/partition_data6_list/train_gt_mean.npy', root_dir + 'partition/partition_data6_list/train_gt_std.npy')
    elif args.model_arch.lower() == 'neumann':
        model = NEUMANN(root_dir + 'partition/partition_data6_list/train_gt_mean.npy', root_dir + 'partition/partition_data6_list/train_gt_std.npy')
    else:
        raise ValueError('Unknown model arch type: ' + args.model_arch.lower())
        
    return model

def chooseLoss(args, option=0):
    
    loss_fn = None
    if args.model_arch.lower() == 'lpcnn' and option == 0:
        loss_fn = nn.MSELoss()
    elif (args.model_arch.lower() == 'lpcnn' or args.model_arch.lower() == 'lpcnn_unet' or args.model_arch.lower() == 'lpcnn_resunet') and option == 1:
        loss_fn = nn.L1Loss()
    elif args.model_arch.lower() == 'lpcnn' and option == 2:
        loss_fn = nn.L1Loss(reduction='none')
    elif args.model_arch.lower() == 'lpcnn_indep' and option == 0:
        loss_fn = nn.MSELoss()
    elif args.model_arch.lower() == 'lpcnn_indep' and option == 1:
        loss_fn = nn.L1Loss()
    elif args.model_arch.lower() == 'lpcnn_indep' and option == 2:
        loss_fn = nn.L1Loss(reduction='none')
    elif args.model_arch.lower() == 'vdsrr':
        loss_fn = nn.L1Loss()
    elif args.model_arch.lower() == 'neumann' and option == 0:
        loss_fn = nn.MSELoss()
    elif args.model_arch.lower() == 'neumann' and option == 1:
        loss_fn = nn.L1Loss()
    elif args.model_arch.lower() == 'neumann' and option == 2:
        loss_fn = nn.L1Loss(reduction='none')

    else:
        raise ValueError('Unsupported loss function')
        
    return loss_fn

def chooseOptimizer(model, args):
    
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    elif args.optimizer == 'custom':
        pass
    else:
        raise ValueError('Unsupported optimizer: ' + args.optimizer)

    return optimizer
    
def sti_psnr(gt, input_data, mask, root_dir, roi=True, ani=False):

    if roi and ani:
        mask_6 = np.repeat(mask.astype(int), 3, axis=3)
    elif roi:
        mask_6 = np.repeat(mask.astype(int), 6, axis=3)

    path = root_dir + '/partition/partition_data6_list/'

    if ani:
        max_val = 0.03
        min_val = -0.03
    else:
        max_val = np.load(path + 'train_val_gt_max_val.npy')
        min_val = np.load(path + 'train_val_gt_min_val.npy')

    mod_input = np.copy(input_data)
    mod_input[mod_input < min_val] = min_val
    mod_input[mod_input > max_val] = max_val
    
    if roi:
        psnr_value = psnr(gt[mask_6==1], mod_input[mask_6==1], data_range=max_val - min_val)
    else:
        psnr_value = psnr(gt, mod_input, data_range=max_val - min_val)
    
    return psnr_value

def sti_ssim(gt, input_data, mask, root_dir):

    path = root_dir + '/partition/partition_data6_list/'
    max_vec = np.load(path + 'train_val_gt_max_vec.npy')
    min_vec = np.load(path + 'train_val_gt_min_vec.npy')
    
    mod_input = np.copy(input_data)
    
    for i in range(6):
        mod_input[:, :, :, i][mod_input[:, :, :, i] < min_vec[i]] = min_vec[i]
        mod_input[:, :, :, i][mod_input[:, :, :, i] > max_vec[i]] = max_vec[i]

    new_gt = (gt - min_vec) / (max_vec - min_vec)
    new_input = (mod_input - min_vec) / (max_vec - min_vec)
    
    ssim_value = ssim(new_gt, new_input, multichannel=True, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

    return ssim_value

def sti_mse(gt, input_data, mask, roi=True):

    if roi:
        total = np.sum(mask)
    else:
        (x, y, z, _) = gt.shape
        total = x * y * z
    
    mse = np.sum(np.square(gt - input_data)) / total / 6

    return mse

def sti_vec(gt, input_data, ani_mask, thr=0):

    bool_mask = ani_mask > thr #0.01   

    matrix_gt = transform_matrix(gt[bool_mask])
    matrix_input = transform_matrix(input_data[bool_mask])
    
    _, gt_vec = np.linalg.eigh(matrix_gt)
    _, input_vec = np.linalg.eigh(matrix_input)
    
    total, _, _ = gt_vec.shape

    #gt_vec = gt_vec * ani_mask[bool_mask]
    #input_vec = input_vec * ani_mask[bool_mask]

    error = 1 - (np.sum(np.absolute(np.sum(gt_vec[:, :, 2] * input_vec[:, :, 2], axis=1))) / total)
    #error = 3 - (np.sum(np.absolute(np.sum(gt_vec[:, :, 0] * input_vec[:, :, 0], axis=1))) + np.sum(np.absolute(np.sum(gt_vec[:, :, 1] * input_vec[:, :, 1], axis=1))) + np.sum(np.absolute(np.sum(gt_vec[:, :, 2] * input_vec[:, :, 2], axis=1)))) / total

    return error

def sti_psnr_torch(gt, input_data, mask, root_dir, roi=True, ani=False):

    batch_size, _, _, _, _ = gt.shape

    if roi and ani:
        mask_6 = mask.expand(-1, 3, -1, -1, -1)
    elif roi:
        mask_6 = mask.expand(-1, 6, -1, -1, -1)

    path = root_dir + '/partition/partition_data6_list/'

    if ani:
        max_val = 0.03
        min_val = -0.03
    else:
        max_val = torch.from_numpy(np.load(path + 'train_val_gt_max_val.npy')).to(input_data.device, dtype=torch.float)
        min_val = torch.from_numpy(np.load(path + 'train_val_gt_min_val.npy')).to(input_data.device, dtype=torch.float)

    mod_input = input_data.clone()
    mod_input[mod_input < min_val] = min_val
    mod_input[mod_input > max_val] = max_val
    
    diff = gt - mod_input
    sq_diff = diff * diff

    d_range = max_val - min_val
    
    psnr_value = 0

    for i in range(batch_size):

        single_sq_diff = sq_diff[i, :, :, :, :]
        single_mask_6 = mask_6[i, :, :, :, :]
        if roi:
            mse = torch.mean(single_sq_diff[single_mask_6==1])
        else:
            mse = torch.mean(sq_diff[i, :, :, :, :])

        psnr_value += 10 * torch.log10((d_range * d_range) / mse)
    
    return psnr_value

def sti_ssim_torch(gt, input_data, mask, root_dir):

    batch_size, _, _, _, _ = gt.shape

    path = root_dir + '/partition/partition_data6_list/'
    max_vec = torch.from_numpy(np.load(path + 'train_val_gt_max_vec.npy')).to(input_data.device, dtype=torch.float).view(1, -1, 1, 1, 1)
    min_vec = torch.from_numpy(np.load(path + 'train_val_gt_min_vec.npy')).to(input_data.device, dtype=torch.float).view(1, -1, 1, 1, 1)
    
    mod_input = input_data.clone()
    
    for i in range(6):
        mod_input[:, i, :, :, :][mod_input[:, i, :, :, :] < min_vec[0, i, 0, 0, 0]] = min_vec[0, i, 0, 0, 0]
        mod_input[:, i, :, :, :][mod_input[:, i, :, :, :] > max_vec[0, i, 0, 0, 0]] = max_vec[0, i, 0, 0, 0]

    new_gt = (gt - min_vec) / (max_vec - min_vec)
    new_input = (mod_input - min_vec) / (max_vec - min_vec)
    
    ssim_value = 0

    for i in range(batch_size):
        ssim_value += ssim3d(new_gt[i, :, :, :, :].unsqueeze(0), new_input[i, :, :, :, :].unsqueeze(0), dynamic_range=1, sigma=1.5)

    return ssim_value

def sti_mse_torch(gt, input_data, mask, roi=True):

    batch_size, _, _, _, _ = gt.shape

    mse = 0

    diff = gt - input_data
    sq_diff = diff * diff

    for i in range(batch_size):
        if roi:
            total = torch.sum(mask[i, :, :, :, :])
        else:
            (_, _, x, y, z) = gt.shape
            total = x * y * z
    
        mse += torch.sum(sq_diff[i, :, :, :, :]) / total / 6

    return mse

def sti_vec_torch(gt, input_data, ani_mask, thr=0):

    batch_size, _, _, _, _ = gt.shape

    bool_mask = ani_mask > thr #0.01   

    device = torch.device('cpu')

    matrix_gt = transform_matrix_torch(gt).to(device)
    matrix_input = transform_matrix_torch(input_data).to(device)
    
    bool_mask = bool_mask.to(device)

    error = 0

    for i in range(batch_size):
        
        _, gt_eigvec = torch.symeig(matrix_gt[i, :, :, :, :, :][bool_mask[i, :, :, :]], True)
        _, input_eigvec = torch.symeig(matrix_input[i, :, :, :, :, :][bool_mask[i, :, :, :]], True)
    
        total, _, _ = gt_eigvec.shape

        gt_eigvec = gt_eigvec.to(gt.device)
        input_eigvec = input_eigvec.to(gt.device)

        error += 1 - (torch.sum(torch.abs(torch.sum(gt_eigvec[:, :, 2] * input_eigvec[:, :, 2], axis=1))) / total)

    #error = 1 - (np.sum(np.absolute(np.sum(gt_vec[:, :, 2] * input_vec[:, :, 2], axis=1))) / total)
    #error = 3 - (np.sum(np.absolute(np.sum(gt_vec[:, :, 0] * input_vec[:, :, 0], axis=1))) + np.sum(np.absolute(np.sum(gt_vec[:, :, 1] * input_vec[:, :, 1], axis=1))) + np.sum(np.absolute(np.sum(gt_vec[:, :, 2] * input_vec[:, :, 2], axis=1)))) / total
    error = error / batch_size

    return error


def sti_wpsnr_torch(gt, input_data, mask):
    """

    """
    if torch.isnan(gt).any():
        return np.NaN
    batch_size, _, _, _, _ = gt.shape

    device = torch.device('cpu')

    matrix_gt = transform_matrix_torch(gt).to(device)
    matrix_input = transform_matrix_torch(input_data).to(device)
    
    bool_mask = (mask==1).to(device)

    wpsnr = 0
    
    data_max = 0.08
    data_min = -0.08

    for i in range(batch_size):
        gt_eigval, gt_eigvec = torch.symeig(matrix_gt[i, :, :, :, :, :][bool_mask[i, 0, :, :, :]], True)
        input_eigval, input_eigvec = torch.symeig(matrix_input[i, :, :, :, :, :][bool_mask[i, 0, :, :, :]], True)
        
        gt_ani = gt_eigval[:, 2] - (gt_eigval[:, 1] + gt_eigval[:, 0]) / 2
        input_ani = input_eigval[:, 2] - (input_eigval[:, 1] + input_eigval[:, 0]) / 2
        
        total, _, _ = gt_eigvec.shape
        
        mod_input = input_ani.unsqueeze(1) * input_eigvec[:,:,2]
        mod_gt = gt_ani.unsqueeze(1) * gt_eigvec[:,:,2]
        
        mod_input = torch.clamp(mod_input, data_min, data_max)
        
        data_range = data_max - data_min # max{all data} - min{all data}
        max_sig_power = (data_range)**2
        noise_power = torch.mean((mod_input-mod_gt)**2)
        print('np: {:.8f}'.format(noise_power))
        wpsnr = wpsnr + 10*np.log10(max_sig_power/noise_power)
        

        
    wpsnr = wpsnr / batch_size

    return wpsnr


def transform_matrix(tensor_data):
    
    # tensor as a 4D file in this order: Dxx,Dxy,Dxz,Dyy,Dyz,Dzz
    (N, channel) = tensor_data.shape
    
    matrix_data = np.zeros((N, 3, 3))
    
    matrix_data[:, 0, 0] = tensor_data[:, 0]
    matrix_data[:, 0, 1] = tensor_data[:, 1]
    matrix_data[:, 0, 2] = tensor_data[:, 2]
    matrix_data[:, 1, 0] = tensor_data[:, 1]
    matrix_data[:, 1, 1] = tensor_data[:, 3]
    matrix_data[:, 1, 2] = tensor_data[:, 4]
    matrix_data[:, 2, 0] = tensor_data[:, 2]
    matrix_data[:, 2, 1] = tensor_data[:, 4]
    matrix_data[:, 2, 2] = tensor_data[:, 5]

    return matrix_data

def transform_matrix_torch(tensor_data):
    
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
