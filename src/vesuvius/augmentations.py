import numpy as np
import random

class RandomFlip:
    def __init__(self, axes=(0,1,2), p=0.5):
        self.axes = axes
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            axis = random.choice(self.axes)
            sample['image'] = np.flip(sample['image'], axis=axis+0).copy()
            if 'label' in sample:
                sample['label'] = np.flip(sample['label'], axis=axis).copy()
        return sample

class RandomRotate90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            k = random.randint(0,3)
            sample['image'] = np.rot90(sample['image'], k, axes=(1,2)).copy()
            if 'label' in sample:
                sample['label'] = np.rot90(sample['label'], k, axes=(1,2)).copy()
        return sample

class IntensityShift:
    def __init__(self, scale=0.1, p=0.5):
        self.scale = scale
        self.p = p
    def __call__(self, sample):
        if random.random() < self.p:
            noise = np.random.normal(0, self.scale, size=sample['image'].shape).astype('float32')
            sample['image'] = sample['image'] + noise
            sample['image'] = sample['image'].astype('float32')
        return sample

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
