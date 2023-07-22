import os
import sys
import cv2
import PIL
import random
import openslide
import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage as nd
from torch.utils.data import Dataset

class WSIPatchDataset(Dataset):

    def __init__(self, slide, prior, level_sample, level_ckpt, args, file,
                 image_size=256, normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._prior = prior
        self._level_sample = level_sample
        self._level_ckpt = level_ckpt
        self._args = args
        self._file = file
        self._patch_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._image_size = tuple([int(i / 2**self._level_ckpt) for i in self._slide.level_dimensions[0]])
        self.first_stage_map, self.dist_from_edge, self.nearest_bg_coord, self.feature_region_conf = self._prior
        
        self._POI = (self.dist_from_edge == 1)
        # self._POI = (self.dist_from_edge >= 1)
        
        self._resolution = 2 ** (self._level_sample - self._level_ckpt)
        self._X_idcs, self._Y_idcs = np.where(self._POI)
        self._idcs_num = len(self._X_idcs)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        x_center = int(x_mask * self._resolution)
        y_center = int(y_mask * self._resolution)

        patch_size = self._patch_size

        x = int((x_center - patch_size // 2) * self._slide.level_downsamples[self._level_ckpt])
        y = int((y_center - patch_size // 2) * self._slide.level_downsamples[self._level_ckpt])
        
        img = self._slide.read_region(
            (x, y), self._level_ckpt, (patch_size, patch_size)).convert('RGB')
        
        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self._rotate == 'ROTATE_90':
            img = img.transpose(PIL.Image.ROTATE_90)

        if self._rotate == 'ROTATE_180':
            img = img.transpose(PIL.Image.ROTATE_180)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(PIL.Image.ROTATE_270)

            # PIL image:   H x W x C
            # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 1, 0))
        
        if self._normalize:
            img = (img - 128.0) / 128.0
        
        left = max(x_center - patch_size // 2, 0)
        right = min(x_center + patch_size // 2, self._image_size[0])
        top = max(y_center - patch_size // 2, 0)
        bot = min(y_center + patch_size // 2, self._image_size[1])
        
        l = left - (x_center - patch_size // 2)
        r = patch_size + right - (x_center + patch_size // 2)
        t = top - (y_center - patch_size // 2)
        b = patch_size + bot - (y_center + patch_size // 2)
        return (img, (left, top, right, bot), (l, t, r, b, patch_size))
        