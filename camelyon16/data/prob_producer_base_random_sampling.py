import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL

class WSIPatchDataset(Dataset):

    def __init__(self, slide, prior, level_sample, level_ckpt, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        first_stage_map = prior[0]
        dist_from_edge = prior[1]
        nearest_bg_coord = prior[2]
        
        # self._tissue = dist_from_edge > 0
        
        self._tissue = dist_from_edge == 1
        self._tissue[nearest_bg_coord[:, self._tissue][0], nearest_bg_coord[:, self._tissue][1]] = True
        
        self._distance = dist_from_edge * 2 ** (level_sample - level_ckpt)
        
        # import cv2
        # img_rgb = slide.read_region((0, 0), level_sample, \
        #             tuple([int(i/2**level_sample) for i in slide.level_dimensions[0]])).convert('RGB')
        # img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        # map_rgb = cv2.applyColorMap(first_stage_map.astype(np.uint8), cv2.COLORMAP_JET)
        # map_rgb = cv2.cvtColor(map_rgb, cv2.COLOR_BGR2RGB)
        # heat_img = cv2.addWeighted(map_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        # heat_img[self._tissue.transpose(1, 0), :] = [0, 255, 0]
        # cv2.imwrite('/media/ps/passport2/hhy/camelyon16/test/dens_map_sampling_2s_l6/model_distance_l1/heat.png', heat_img)
         
        self._level_sample = level_sample
        self._level_ckpt = level_ckpt
        self._patch_size = image_size
        self._image_size = tuple([int(i / 2**level_ckpt) for i in slide.level_dimensions[0]])
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self): 
        self._resolution = 2 ** (self._level_sample - self._level_ckpt)
        self._X_idcs, self._Y_idcs = np.where(self._tissue)
        self._idcs_num = len(self._X_idcs)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        x_center = int(x_mask * self._resolution)
        y_center = int(y_mask * self._resolution)

        # patch_prob = self._saliency[x_center-128:x_center+128, y_center-128:y_center+128].mean() / 255
        # patch_size = int(256 / 2** (np.abs(patch_prob - 0.5) * 5)) // 2 * 2
        patch_size = self._patch_size
        
        x = int((x_center - patch_size / 2) * self._slide.level_downsamples[self._level_ckpt])
        y = int((y_center - patch_size / 2) * self._slide.level_downsamples[self._level_ckpt])
        
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
        r = l + right - left
        t = top - (y_center - patch_size // 2)
        b = t + bot - top
        return (img, (left, top, right, bot), (l, t, r, b))
        