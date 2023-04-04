import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL

class WSIPatchDataset(Dataset):

    def __init__(self, slide, level, image_size=256, crop_size=224,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._level = level
        self._img = slide.read_region((0, 0), level,
                        tuple([int(i / 2**level) for i in slide.level_dimensions[0]])).convert('RGB')
        self._img = self._img.resize(slide.level_dimensions[level])
        self._image_size = image_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        X_slide, Y_slide = self._slide.level_dimensions[self._level]
        
        X_idcs_num = X_slide // self._image_size
        Y_idcs_num = Y_slide // self._image_size
        
        X_mesh_idcs = np.arange(X_idcs_num) * self._image_size
        Y_mesh_idcs = np.arange(Y_idcs_num) * self._image_size

        if X_slide % self._image_size != 0:
            X_mesh_idcs = np.arange(X_idcs_num + 1) * self._image_size
        if Y_slide % self._image_size != 0:
            Y_mesh_idcs = np.arange(Y_idcs_num + 1) * self._image_size

        X_idcs_mesh, Y_idcs_mesh = np.meshgrid(X_mesh_idcs, Y_mesh_idcs)
        self._X_idcs = X_idcs_mesh.flatten()
        self._Y_idcs = Y_idcs_mesh.flatten()

        self._idcs_num = len(self._X_idcs)
        # self._idcs_num = 1

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x, y = self._X_idcs[idx], self._Y_idcs[idx]

        x_size = min(self._image_size, self._slide.level_dimensions[self._level][0] - x)
        y_size = min(self._image_size, self._slide.level_dimensions[self._level][1] - y)
        img = self._img.crop((x, y, x + x_size, y + y_size))

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
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return (img, (x, y, x + x_size, y + y_size))

    # def __getitem__(self, idx):
    #     img = self._img

    #     if self._flip == 'FLIP_LEFT_RIGHT':
    #         img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    #     if self._rotate == 'ROTATE_90':
    #         img = img.transpose(PIL.Image.ROTATE_90)

    #     if self._rotate == 'ROTATE_180':
    #         img = img.transpose(PIL.Image.ROTATE_180)

    #     if self._rotate == 'ROTATE_270':
    #         img = img.transpose(PIL.Image.ROTATE_270)

    #         # PIL image:   H x W x C
    #         # torch image: C X H X W
    #     img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

    #     if self._normalize:
    #         img = (img - 128.0) / 128.0

    #     return (img, (0, 0, self._slide.level_dimensions[self._level][0], self._slide.level_dimensions[self._level][1]))