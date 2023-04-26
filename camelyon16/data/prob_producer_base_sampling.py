import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL

class WSIPatchDataset(Dataset):

    def __init__(self, slide, tissue, sample_level, ckpt_level, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._tissue = tissue
        self._sample_level = sample_level
        self._ckpt_level = ckpt_level
        self._patch_size = image_size
        self._image_size = tuple([int(i / 2**ckpt_level) for i in slide.level_dimensions[0]])
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self): 
        self._resolution = 2 ** (self._sample_level - self._ckpt_level)
        self._X_idcs, self._Y_idcs = np.where(self._tissue)
        self._idcs_num = len(self._X_idcs)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        x_center = int(x_mask * self._resolution)
        y_center = int(y_mask * self._resolution)

        x = int((x_center - self._patch_size / 2) * self._slide.level_downsamples[self._ckpt_level])
        y = int((y_center - self._patch_size / 2) * self._slide.level_downsamples[self._ckpt_level])
        
        img = self._slide.read_region(
            (x, y), self._ckpt_level, (self._patch_size, self._patch_size)).convert('RGB')
        
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
        
        left = max(x_center - self._patch_size // 2, 0)
        right = min(x_center + self._patch_size // 2, self._image_size[0])
        top = max(y_center - self._patch_size // 2, 0)
        bot = min(y_center + self._patch_size // 2, self._image_size[1])
        
        l = left - (x_center - self._patch_size // 2)
        r = l + right - left
        t = top - (y_center - self._patch_size // 2)
        b = t + bot - top
        return (img, (left, top, right, bot), (l, t, r, b))
        