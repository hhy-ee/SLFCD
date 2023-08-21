import cv2
import PIL
import numpy as np
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
        self.prior_map, self.render_seq, self.origin_cluster, self.moved_cluster, self.bin_size = self._prior
        
        self._canvas_size = self.bin_size
        
        self._idcs_num = len( self.render_seq)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        canvas = np.zeros((3, self._canvas_size[idx], self._canvas_size[idx]), dtype=np.float32)
        
        img_box, canvas_box, patch_box = [], [], []
        for p_idx in self.render_seq[idx]:
            box = self.origin_cluster[idx][p_idx]
            box[2], box[3] = max(box[2], box[0] + 1), max(box[3], box[1] + 1)
            x_mask, y_mask, x_size, y_size = box[0], box[1], box[2]-box[0], box[3]-box[1]
            
            x = int(x_mask * self._slide.level_downsamples[self._level_ckpt])
            y = int(y_mask * self._slide.level_downsamples[self._level_ckpt])
            
            moved_box = self.moved_cluster[idx][p_idx]
            x_start, y_start, x_end, y_end = moved_box
            x_patch_size = max(1, int(x_end - x_start))
            y_patch_size = max(1, int(y_end - y_start))
            x_start, y_start = int(x_start), int(y_start), 
            x_end, y_end = x_start + x_patch_size, y_start + y_patch_size
            
            img = self._slide.read_region(
                    (x, y), self._level_ckpt, (x_size, y_size)).convert('RGB')
            img = img.resize((x_patch_size, y_patch_size))
            
            if self._flip == 'FLIP_LEFT_RIGHT':
                    img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

            if self._rotate == 'ROTATE_90':
                img = img.transpose(PIL.Image.ROTATE_90)

            if self._rotate == 'ROTATE_180':
                img = img.transpose(PIL.Image.ROTATE_180)

            if self._rotate == 'ROTATE_270':
                img = img.transpose(PIL.Image.ROTATE_270)

            # PIL image:   H x W x C, torch image: C X H X W
            img = np.array(img, dtype=np.float32).transpose((2, 1, 0))
            
            canvas[:, x_start: x_end, y_start: y_end] = img
            
            left, top, right, bot = box
            img_box.append((left, top, right, bot, x_size, y_size))
            canvas_box.append((x_start, y_start, x_end, y_end))
            
        if self._normalize:
            canvas = (canvas - 128.0) / 128.0

        return (canvas, img_box, canvas_box)