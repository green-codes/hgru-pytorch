"""
Data generators and related stuff
"""

import os
import scipy.io
import numpy as np
import imageio
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader

class PathfinderDataset(Dataset):

    def __init__(self, data_root, transform=None):
        
        self.transform = transform
        
        """ load only paths to images """
        data_root += '/' if not data_root.endswith('/') else ''
        self._x_files = []
        self._y_arr_all = []

        # positive samples
        x_path = data_root + "curv_baseline/imgs/"
        dirs = [d for d in os.listdir(x_path) if os.path.isdir(os.path.join(x_path,d))]
        for d in dirs:
            for f in self._get_filenames(x_path+d, '.png'):
                self._x_files += ["{}{}/{}".format(x_path,d,f)]
                self._y_arr_all += [1]

        # negative samples
        x_path = data_root + "curv_baseline_neg/imgs/"
        dirs = [d for d in os.listdir(x_path) if os.path.isdir(os.path.join(x_path,d))]
        for d in dirs:
            for f in self._get_filenames(x_path+d, '.png'):
                self._x_files += ["{}{}/{}".format(x_path,d,f)]
                self._y_arr_all += [0]

        self._x_files = np.array(self._x_files)
        self._y_arr_all = np.array(self._y_arr_all)
        self._nsamples = len(self._x_files)
        self._iter_idx = 0

    def __len__(self):
        return self._nsamples

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        x = Image.open(self._x_files[index]).convert('L')
        y = np.array(self._y_arr_all[index])
        ret = {'image':x, 'label':y}
        if self.transform is not None:
            ret = self.transform(ret)
        return ret

    def _get_filenames(self, path, extension):
        return [x for x in os.listdir(path) if x.endswith(extension)]
