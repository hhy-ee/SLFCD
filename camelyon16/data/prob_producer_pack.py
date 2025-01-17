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
        self.prior_map, self.assign_per_img = self._prior

        self._X_idcs = [i['keep'][0] for i in self.assign_per_img]
        self._Y_idcs = [i['keep'][1] for i in self.assign_per_img]
        self._X_size = [i['keep'][2] for i in self.assign_per_img]
        self._Y_size = [i['keep'][2] for i in self.assign_per_img]
        
        if self._args.random_shuffle:
            shuffle_idx = np.random.permutation(list(range(len(self._X_idcs))))
            self._X_idcs = np.array(self._X_idcs)[shuffle_idx]
            self._Y_idcs = np.array(self._Y_idcs)[shuffle_idx]
            self._X_size = np.array(self._X_size)[shuffle_idx]
            self._Y_size = np.array(self._Y_size)[shuffle_idx]
        
        self._canvas_size = self._args.canvas_size
        
        self._patch_per_side = int(self._canvas_size / self._patch_size)
        
        self._interval = (self._canvas_size - self._patch_size * self._patch_per_side) \
                        // (self._patch_per_side - 1) if self._patch_per_side != 1 else 0
        
        self._idcs_num = int(np.ceil(len(self._X_idcs) / self._patch_per_side**2))

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        canvas = np.zeros((3, self._canvas_size, self._canvas_size), dtype=np.float32)
        
        X_idcs = self._X_idcs[idx * self._patch_per_side**2 : (idx + 1) * self._patch_per_side**2]
        Y_idcs = self._Y_idcs[idx * self._patch_per_side**2 : (idx + 1) * self._patch_per_side**2]
        X_size = self._X_size[idx * self._patch_per_side**2 : (idx + 1) * self._patch_per_side**2]
        Y_size = self._Y_size[idx * self._patch_per_side**2 : (idx + 1) * self._patch_per_side**2]
        
        img_box, canvas_box, patch_box = [], [], []
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                if self._patch_per_side * x_idx + y_idx >= len(X_idcs):
                    continue
                
                x_mask = X_idcs[self._patch_per_side * x_idx + y_idx]
                y_mask = Y_idcs[self._patch_per_side * x_idx + y_idx]
                x_size = X_size[self._patch_per_side * x_idx + y_idx]
                y_size = Y_size[self._patch_per_side * x_idx + y_idx]
                
                x_center = int(x_mask + x_size // 2)
                y_center = int(y_mask + y_size // 2)
                
                if self._args.pack_mode == 'crop':
                    if x_size > self._patch_size or y_size > self._patch_size:
                        x_mask = x_center - self._patch_size // 2
                        y_mask = y_center - self._patch_size // 2
                        x_size, y_size = self._patch_size, self._patch_size

                x = int(x_mask * self._slide.level_downsamples[self._level_ckpt])
                y = int(y_mask * self._slide.level_downsamples[self._level_ckpt])
        
                img = self._slide.read_region(
                    (x, y), self._level_ckpt, (x_size, y_size)).convert('RGB')
                if self._args.pack_mode == 'resize':
                    img = img.resize((self._patch_size, self._patch_size))
                
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
                
                x_start = x_idx * (self._patch_size + self._interval)
                y_start = y_idx * (self._patch_size + self._interval)
                if self._args.pack_mode == 'resize':
                    x_end = x_start + self._patch_size
                    y_end = y_start + self._patch_size
                else:
                    x_end = x_start + x_size
                    y_end = y_start + y_size
                canvas[:, x_start: x_end, y_start: y_end] = img
        
                left = max(x_center - x_size // 2, 0)
                right = min(x_center + x_size // 2, self._image_size[0])
                top = max(y_center - y_size // 2, 0)
                bot = min(y_center + y_size // 2, self._image_size[1])
                
                l = left - (x_center - x_size // 2)
                r = x_size + right - (x_center + x_size // 2)
                t = top - (y_center - y_size // 2)
                b = y_size + bot - (y_center + y_size // 2)
                
                img_box.append((left, top, right, bot))
                canvas_box.append((x_start, y_start, x_end, y_end))
                patch_box.append((l, t, r, b, x_size, y_size))
                
        if self._normalize:
            canvas = (canvas - 128.0) / 128.0
                
        return (canvas, img_box, canvas_box, patch_box)