import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import random
import pandas as pd

class VesuviusDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 image_key='image_path',
                 label_key='label_path',
                 mode='train',
                 patch_size=(128,128,128),
                 transforms=None,
                 preload=False,
                 cache_rate=0.0):
        self.df = df.reset_index(drop=True)
        self.image_key = image_key
        self.label_key = label_key
        self.mode = mode
        self.patch_size = tuple(patch_size)
        self.transforms = transforms
        self.preload = preload
        self.cache_rate = cache_rate

        self._cache = {}
        if self.preload:
            for idx, r in self.df.iterrows():
                img = self._read_tiff(r[image_key])
                lbl = None
                if label_key in r and not pd.isna(r[label_key]):
                    lbl = self._read_tiff(r[label_key])
                self._cache[idx] = (img, lbl)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _read_tiff(path):
        arr = tifffile.imread(str(path))
        return arr.astype(np.float32)

    def _random_crop(self, image, label=None):
        D, H, W = image.shape
        pD, pH, pW = self.patch_size
        if D <= pD or H <= pH or W <= pW:
            pad_d = max(0, pD - D)
            pad_h = max(0, pH - H)
            pad_w = max(0, pW - W)
            image = np.pad(image,
                           ((0, pad_d), (0, pad_h), (0, pad_w)),
                           mode='constant', constant_values=0)
            if label is not None:
                label = np.pad(label,
                               ((0, pad_d), (0, pad_h), (0, pad_w)),
                               mode='constant', constant_values=0)
            D, H, W = image.shape

        z = random.randint(0, D - pD)
        y = random.randint(0, H - pH)
        x = random.randint(0, W - pW)
        img_patch = image[z:z+pD, y:y+pH, x:x+pW]
        lbl_patch = None
        if label is not None:
            lbl_patch = label[z:z+pD, y:y+pH, x:x+pW]
        return img_patch, lbl_patch

    def __getitem__(self, idx):
        if idx in self._cache:
            img, lbl = self._cache[idx]
        else:
            row = self.df.iloc[idx]
            img = self._read_tiff(row[self.image_key])
            lbl = None
            if self.label_key in row and not pd.isna(row[self.label_key]):
                lbl = self._read_tiff(row[self.label_key])
            if self.cache_rate > 0 and np.random.rand() < self.cache_rate:
                self._cache[idx] = (img, lbl)

        if self.mode == 'train':
            img, lbl = self._random_crop(img, lbl)
        else:
            D, H, W = img.shape
            pD, pH, pW = self.patch_size
            z = (D - pD)//2 if D > pD else 0
            y = (H - pH)//2 if H > pH else 0
            x = (W - pW)//2 if W > pW else 0
            img = img[z:z+pD, y:y+pH, x:x+pW]
            if lbl is not None:
                lbl = lbl[z:z+pD, y:y+pH, x:x+pW]

        sample = {'image': img[None, ...].astype('float32')}
        if lbl is not None:
            sample['label'] = lbl.astype('int64')
        if self.transforms:
            sample = self.transforms(sample)
        sample['image'] = torch.from_numpy(sample['image'])
        if 'label' in sample:
            sample['label'] = torch.from_numpy(sample['label'])
        return sample
