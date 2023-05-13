import os
import cv2
import PIL
import numpy as np
from torch.utils.data import Dataset

class WSIPatchDataset(Dataset):

    def __init__(self, slide, level_ckpt, assign, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._level_ckpt = level_ckpt
        self._assign = assign
        self._patch_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self.image_batch = []
        self.boxes = []
        self.patches = []
        for assign in self._assign:   
            o_x1, o_x2, o_y1, o_y2 = assign[1][0], assign[2][0], assign[1][1], assign[2][1]
            o_s_x1 = int(o_x1 * self._slide.level_downsamples[self._level_ckpt])
            o_s_y1 = int(o_y1 * self._slide.level_downsamples[self._level_ckpt])
            w = o_x2 - o_x1
            h = o_y2 - o_y1
            
            box = [o_x1, o_y1, o_x2, o_y2]
            patch = [o_s_x1, o_s_y1, w, h]
            
            self.boxes.append(box)
            self.patches.append(patch)
            
        self._idcs_num = len(self.patches)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        box = self.boxes[idx]
        patch = self.patches[idx]
        o_s_x1, o_s_y1, w, h = patch
        
        img = self._slide.read_region((o_s_x1, o_s_y1), self._level_ckpt, (w, h)).convert('RGB')

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
        img = np.asarray(img).transpose((1, 0, 2))
        resize = False
        if w > 2048 or h >2048:
            img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_CUBIC)
            resize = True
        img = img.astype("float32").transpose(2, 0, 1)

        if self._normalize:
            img = (img - 128.0) / 128.0

        return (img, box, resize)