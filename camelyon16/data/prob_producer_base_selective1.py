import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
from PIL import ImageDraw

class WSIPatchDataset(Dataset):

    def __init__(self, slide, candidate, level, overlap, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._candidate = candidate
        self._level = level
        self._overlap = overlap
        self._img = slide.read_region((0, 0), level,
                        tuple([int(i / 2**level) for i in slide.level_dimensions[0]])).convert('RGB')
        self._img = self._img.resize(slide.level_dimensions[level])

        # img = self._img
        # img_draw = ImageDraw.ImageDraw(img) 
        # for bbox in self._candidate_new:
        #     img_draw.rectangle((tuple(bbox[0]), tuple(bbox[1])), fill=None, outline='blue', width=5)
        # img.save('/media/ps/passport2/hhy/camelyon16/training/0.png')

        self._image_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._candidate_new = []
        for patch in self._candidate:
            (left, top) , (right, bot) = patch[0], patch[1]
            if (right - left) > self._image_size or (bot - top) > self._image_size:
                X_idcs_num = (right - left) // self._image_size
                Y_idcs_num = (bot - top) // self._image_size

                X_mesh_idcs = np.arange(X_idcs_num) * self._image_size
                Y_mesh_idcs = np.arange(Y_idcs_num) * self._image_size
                
                if (right - left) % self._image_size != 0:
                    X_mesh_idcs = np.arange(X_idcs_num + 1) * self._image_size
                if (bot - top) % self._image_size != 0:
                    Y_mesh_idcs = np.arange(Y_idcs_num + 1) * self._image_size

                for i in X_mesh_idcs:
                    for j in Y_mesh_idcs:
                        self._candidate_new.append([[i+left, j+top], \
                                        [min(i+left+self._image_size, right), min(j+top+self._image_size, bot)]])
            else:
                self._candidate_new.append(patch)

        self._idcs_num = len(self._candidate)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        left, top = self._candidate_new[idx][0]
        right, bot = self._candidate_new[idx][1]
        img = self._img.crop((left, top, right, bot))

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

        return (img, (left, top, right, bot))

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