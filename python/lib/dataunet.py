from __future__ import print_function
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

import lib.render as render
#####################################################
class UnetDataset(Dataset):
    
    def __init__(self, root_dir, ext='jpg',  transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if os.path.isdir(root_dir) is not True:
            raise ValueError('Path {} is not directory'.format(root_dir))

        self.root_dir = root_dir
        self.transform = transform

        self.files = [ f for f in sorted(os.listdir(self.root_dir)) if f.lower().split('.')[-1] == ext.lower() ]
        self.num = len(self.files)
        self.ext = ext

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.files[idx])
        image = io.imread(img_name)
        sample = {'image': blurimage, 'label': np.array([blureg])}

        if self.transform:
            sample = self.transform(sample)

        return sample

####################################################################################################
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, blureg = sample['image'], sample['regressor']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'regressor': blureg}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, blureg = sample['image'], sample['regressor']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image/255.0
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'regressor': torch.from_numpy(blureg).float()}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, regressor = sample['image'], sample['regressor']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'regressor': regressor}