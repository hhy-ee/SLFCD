import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
from PIL import ImageDraw

class WSIPatchDataset(Dataset):

    def __init__(self, slide, tissue, level, candidate, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self.x_scale = slide.level_dimensions[0][0] / 2**level / slide.level_dimensions[level][0]
        self.y_scale = slide.level_dimensions[0][1] / 2**level / slide.level_dimensions[level][1]
        self._image_size = tuple([int(i / 2**level) for i in slide.level_dimensions[0]])
        self._tissue = PIL.Image.fromarray(tissue.transpose()).resize(self._image_size)
        self._level = level
        self._candidate = candidate
        self._patch_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._candidate_new = []
        for patch in self._candidate:
            (left, top) , (right, bot) = patch[0], patch[1]
            left_r = int(left * self.x_scale)
            top_r = int(top * self.y_scale)
            width = right - left + 1
            height = bot -top + 1
            if width > self._patch_size or height > self._patch_size:
                X_idcs_num = width // self._patch_size
                Y_idcs_num = height // self._patch_size

                X_mesh_idcs = np.arange(X_idcs_num) * self._patch_size
                Y_mesh_idcs = np.arange(Y_idcs_num) * self._patch_size
                
                if width % self._patch_size != 0:
                    X_mesh_idcs = np.arange(X_idcs_num + 1) * self._patch_size
                if height % self._patch_size != 0:
                    Y_mesh_idcs = np.arange(Y_idcs_num + 1) * self._patch_size

                for i in X_mesh_idcs:
                    for j in Y_mesh_idcs:
                        self._candidate_new.append([[left_r + i, top_r + j], \
                                        [min(self._patch_size, width - i), \
                                         min(self._patch_size, height - j)]])
            else:
                self._candidate_new.append([[left_r, top_r], [width, height]])

        self._idcs_num = len(self._candidate)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        left, top = self._candidate_new[idx][0]
        width, height = self._candidate_new[idx][1]
        
        img = self._slide.read_region((left, top), self._level, (width, height)).convert('RGB')

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

        return (img, (left, top, left + width, top + height))

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