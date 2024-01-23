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
    Modified from: https://github.com/yulequan/UA-MT/blob/88ed29ad794f877122e542a7fa9505a76fa83515/code/dataloaders/la_heart.py#L162
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
    """Get prediction dataset"""

    if args.dataset == 'sti':

        prediction_set, _, _, _, case = prediction_data

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
    