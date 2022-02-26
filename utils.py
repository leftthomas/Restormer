import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop


def parse_args():
    desc = 'Pytorch Implementation of \'RCDNet: A Model-driven Deep Neural Network for Single Image Rain Removal\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='/home/data')
    parser.add_argument('--data_name', type=str, default='rain100L',
                        choices=['rain100L', 'rain100H', 'rain1400', 'spa'])
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--num_map', type=int, default=32, help='number of rain maps')
    parser.add_argument('--num_channel', type=int, default=32, help='number of channels')
    parser.add_argument('--num_block', type=int, default=4, help='number of res blocks in each ProxNet')
    parser.add_argument('--num_stage', type=int, default=17, help='number of iterative stages')
    parser.add_argument('--num_iter', type=int, default=100, help='epoch of training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of loading images')
    parser.add_argument('--patch_size', type=int, default=64, help='patch size of each image')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--milestone', nargs='+', type=int, default=[25, 50, 75], help='when to decay learning rate')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    # model_file is None means training stage, else means testing stage
    parser.add_argument('--model_file', type=str, default=None, help='path of pre-trained model file')

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_name = args.data_name
        self.save_path = args.save_path
        self.num_map = args.num_map
        self.num_channel = args.num_channel
        self.num_block = args.num_block
        self.num_stage = args.num_stage
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return Config(args)


class RainDataset(Dataset):
    def __init__(self, data_path, data_name, data_type, patch_size=None, length=None):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = data_name, data_type, patch_size
        self.rain_images = sorted(glob.glob('{}/{}/{}/rain/*.png'.format(data_path, data_name, data_type)))
        self.norain_images = sorted(glob.glob('{}/{}/{}/norain/*.png'.format(data_path, data_name, data_type)))
        # make sure the length of training and testing different
        self.num = len(self.rain_images)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        image_name = os.path.basename(self.rain_images[idx % self.num])
        rain = T.pil_to_tensor(Image.open(self.rain_images[idx % self.num])).float()
        norain = T.pil_to_tensor(Image.open(self.norain_images[idx % self.num])).float()

        if self.data_type == 'train':
            i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
        return rain, norain, image_name


def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()
