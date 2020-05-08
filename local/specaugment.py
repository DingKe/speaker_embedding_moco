"""
numpy based SpecAugment.
Reference:
SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
Note: We use the layout of batch_size x timesteps x spec_feat
"""

from __future__ import absolute_import

import concurrent
import concurrent.futures

import copy
import random
import numpy as np
from PIL import Image
from PIL.Image import BICUBIC


class TimeWarp(object):
    def __init__(self, seed=1111, max_workers=20):
        self.seed = seed
        self.random = random.Random()
        self.random.seed(seed)
        self.max_workers = max_workers
        self.executor = None

    @staticmethod
    def warp(x, center, warped):
        left = Image.fromarray(x[:center]).resize((x.shape[1], warped),
                                                  BICUBIC)
        right = Image.fromarray(x[center:]).resize(
            (x.shape[1], len(x) - warped), BICUBIC)
        x[:warped] = left
        x[warped:] = right

    def __call__(self, spec, width=40, inplace=False):
        """
        Args:
            spec (numpy ndarray or list of ndarrays): 
                of size (B, T, F) where the time warping is to be applied.
        Returns:
            spec : corroputed spec with time mask.
        """
        # lazy intialization of executor
        if not self.executor:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers)

        if not inplace:
            spec = copy.deepcopy(spec)

        xs, centers, warpeds = [], [], []
        for x in spec:
            time_steps = x.shape[0]
            freq_bins = x.shape[1]

            if time_steps <= 2 * width:
                continue

            center = random.randrange(width, time_steps - width)
            warped = random.randrange(center - width, center + width) + 1
            xs.append(x)
            centers.append(center)
            warpeds.append(warped)
        self.executor.map(self.warp, zip(xs, centers, warpeds))

        return spec


class TimeMask(object):
    def __init__(self, seed=1111):
        self.seed = seed
        self.random = random.Random()
        self.random.seed(seed)

    def __call__(self,
                 spec,
                 max_width=30,
                 num_masks=2,
                 p=0.2,
                 inplace=False,
                 use_mean=False):
        """
        Args:
            spec (numpy ndarray or list of ndarrays): 
                of size (B, T, F) where the time mask is to be applied.
        Returns:
            spec : corroputed spec with time mask.
        """
        if not inplace:
            spec = copy.deepcopy(spec)

        for x in spec:
            time = x.shape[0]
            max_width = min(max_width, int(p * time))

            filler = x.mean(axis=0) if use_mean else 0
            for i in range(num_masks):
                width = self.random.randrange(0, max_width + 1)
                if width == 0 or time - width <= 0: continue
                start = self.random.randrange(0, time - width)
                end = start + width
                x[start:end, :] = filler

        return spec


class FrequencyMask(object):
    def __init__(self, seed=1111):
        self.seed = seed
        self.random = random.Random()
        self.random.seed(seed)

    def __call__(self,
                 spec,
                 max_width=5,
                 num_masks=2,
                 inplace=False,
                 use_mean=False):
        """
        Args:
            spec (numpy ndarray or list of ndarrays): 
                of size (B, T, F) where the frequency mask is to be applied.
        Returns:
            spec : corrupted spec with frequency mask.
        """
        if not inplace:
            spec = copy.deepcopy(spec)

        for x in spec:
            freq = x.shape[1]

            filler = x.mean(axis=0) if use_mean else 0
            for i in range(num_masks):
                width = self.random.randrange(0, max_width + 1)
                if width == 0 or freq - width <= 0: continue
                start = self.random.randrange(0, freq - width)
                end = start + width
                x[:, start:end] = filler[start:end] if use_mean else 0

        return spec


class SpecAugment(object):
    def __init__(self,
                 seed=1111,
                 warp_width=5,
                 max_freq_width=15,
                 num_freq_masks=1,
                 max_time_width=15,
                 num_time_masks=1,
                 p=0.2,
                 inplace=False,
                 use_mean=False,
                 apply_time_warp=True,
                 max_workers=20):
        self.__dict__.update(locals())
        self.__dict__.pop('self')

        self.time_masker = TimeMask(seed=seed)
        self.freq_masker = FrequencyMask(seed=seed)
        self.time_warper = TimeWarp(seed=seed, max_workers=max_workers)

    def __call__(self, spec):
        # step 1: time warping
        if self.apply_time_warp:
            spec = self.time_warper(spec, self.warp_width, self.inplace)
        # step 2: time masking
        spec = self.time_masker(spec, self.max_time_width, self.num_time_masks,
                                self.p, self.inplace, self.use_mean)
        # step 3: frequency masking
        spec = self.freq_masker(spec, self.max_freq_width, self.num_freq_masks,
                                self.inplace, self.use_mean)
        return spec
