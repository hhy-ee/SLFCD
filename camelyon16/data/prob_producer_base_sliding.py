import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL

class WSIPatchDataset(Dataset):

    def __init__(self, slide, tissue, level, overlap, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._image_size = tuple([int(i / 2**level) for i in slide.level_dimensions[0]])
        self._tissue = PIL.Image.fromarray(tissue.transpose()).resize(self._image_size)
        self._level = level
        self._overlap = overlap
        self._patch_size = image_size
        self._interval = int(self._patch_size * (1-self._overlap))
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        X_slide, Y_slide = self._image_size
        
        X_idcs_num = (X_slide - self._patch_size) // self._interval + 1
        Y_idcs_num = (Y_slide - self._patch_size) // self._interval + 1
        
        X_mesh_idcs = np.arange(X_idcs_num) * self._interval
        Y_mesh_idcs = np.arange(Y_idcs_num) * self._interval

        if (X_slide - self._patch_size) % self._interval != 0:
            X_mesh_idcs = np.arange(X_idcs_num + 1) * self._interval
        if (X_slide - self._patch_size) % self._interval != 0:
            Y_mesh_idcs = np.arange(Y_idcs_num + 1) * self._interval

        X_idcs_mesh, Y_idcs_mesh = np.meshgrid(X_mesh_idcs, Y_mesh_idcs)
        self._X_idcs = X_idcs_mesh.flatten()
        self._Y_idcs = Y_idcs_mesh.flatten()

        self._idcs_num = len(self._X_idcs) 

        # self._idcs_num = 1

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x, y = self._X_idcs[idx], self._Y_idcs[idx]

        x_size = min(self._patch_size, self._image_size[0] - x)
        y_size = min(self._patch_size, self._image_size[1] - y)

        x_crop = int(x * self._slide.level_downsamples[self._level])
        y_crop = int(y * self._slide.level_downsamples[self._level])

        img = self._slide.read_region((x_crop, y_crop), self._level, (self._patch_size, self._patch_size)).convert('RGB')

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

        patch_tissue = self._tissue.crop((x, y, x + self._patch_size, y + self._patch_size))

        if np.asarray(patch_tissue).any():
            return (img, (x, y, x + self._patch_size, y + self._patch_size), (x_size, y_size), True)
        else:
            return (img, (x, y, x + self._patch_size, y + self._patch_size), (x_size, y_size), False)