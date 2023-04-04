import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL

class WSIPatchDataset(Dataset):

    def __init__(self, slide, tissue, level, overlap, image_level, image_size,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._level = level
        self._overlap = overlap
        self._tissue = tissue
        self._image_level = image_level
        self._image_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._img = self._slide.read_region((0, 0), self._level,
                        tuple([int(i / 2**self._level) for i in self._slide.level_dimensions[0]])).convert('RGB')
        self._img = self._img.resize(self._slide.level_dimensions[self._level])

        if not np.log2(self._image_level).is_integer():
            raise Exception('image level is not power of 2: '
                            '{}'.format(self._image_level))
        else:
            self._tissue = PIL.Image.fromarray(self._tissue).\
                resize(tuple([int(i / 2**np.log2(self._image_level)) for i in self._slide.level_dimensions[self._level]]))
        
        self._Y_idcs, self._X_idcs = np.where(self._tissue) 
        
        self._X_idcs = self._X_idcs * 2** (6 - self._level)
        self._Y_idcs = self._Y_idcs * 2** (6 - self._level)

        self._idcs_num = len(self._X_idcs)
        # self._idcs_num = 1

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x, y = self._X_idcs[idx], self._Y_idcs[idx]

        left, right = int(x -(self._image_size - 1) / 2), int(x + (self._image_size + 1) / 2)
        top, bottom = int(y -(self._image_size - 1) / 2), int(y + (self._image_size + 1) / 2)

        x_size = min(self._image_size, self._slide.level_dimensions[self._level][0] - left, right)
        y_size = min(self._image_size, self._slide.level_dimensions[self._level][1] - top, bottom)

        img = self._img.crop((left, top, left + x_size, top + y_size))
        if x_size !=self._image_size or y_size !=self._image_size:
            img_temp = img
            img = PIL.Image.new(img_temp.mode, (self._image_size,)*2)
            left_paste, top_paste = np.abs(min(0, left)), np.abs(min(0, top))
            img.paste(img_temp, (left_paste, top_paste, left_paste + x_size, top_paste + y_size))

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

        return (img, (x-self._image_level//2, y-self._image_level//2, x+self._image_level//2, y+self._image_level//2))

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