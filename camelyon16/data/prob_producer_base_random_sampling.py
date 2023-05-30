import os
import sys
import PIL
import random
import openslide
import numpy as np
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from camelyon16.cluster.utils import NMS

class WSIPatchDataset(Dataset):

    def __init__(self, slide, prior, level_sample, level_ckpt, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._prior = prior
        self._level_sample = level_sample
        self._level_ckpt = level_ckpt
        self._patch_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._image_size = tuple([int(i / 2**self._level_ckpt) for i in self._slide.level_dimensions[0]])
        self.first_stage_map, self.dist_from_edge, self.nearest_bg_coord = self._prior
        # self._POI = (self.dist_from_edge = 1) * (self.first_stage_map / 255 < 0.2)
        self._POI = self.dynamic_sliding_window()
        self._resolution = 2 ** (self._level_sample - self._level_ckpt)
        self._X_idcs, self._Y_idcs = np.where(self._POI)
        self._idcs_num = len(self._X_idcs)

    def dynamic_sliding_window(self, base_patch_size=256, size_threshold=8, nmm_threshold=0.1):
        # my dynamic patch-size algorithm
        self._edge = self.dist_from_edge == 1
        edge_X, edege_Y = np.where(self._edge)
        boxes = []
        for idx in range(0, len(edge_X)):
            x_center, y_center = edge_X[idx], edege_Y[idx]
            ini_size = base_patch_size // 2 ** (self._level_sample - self._level_ckpt)
            l, t = x_center - ini_size // 2, y_center - ini_size // 2
            while self._edge[l: l+ini_size, t: t+ini_size].sum() > size_threshold:
                ini_size = ini_size - 1
                l, t = x_center - ini_size // 2, y_center - ini_size // 2
            scr = self.first_stage_map[x_center, y_center]
            boxes.append([l, t, l+ini_size, t+ini_size, scr, x_center, y_center])
            
        scale = 2 ** (self._level_sample - self._level_ckpt)
        boxes_scale = [[int(i[0] * scale), int(i[1] * scale), int(i[2] * scale), \
                        int(i[3] * scale), i[4], int(i[5] * scale), int(i[6] * scale)] for i in boxes]
        # NMS
        POI = np.zeros(self._edge.shape).astype(int)
        if len(boxes) != 0:
            boxes_scale = np.array(boxes_scale)
            keep_boxes_list, cluster_boxes_dict = NMS(boxes_scale, nmm_threshold)
            boxes_nms = np.array([[i[5]/scale, i[6]/scale, (i[2]-i[0])/scale] for i in keep_boxes_list]).astype(int)
            POI[boxes_nms[:,0], boxes_nms[:,1]] = boxes_nms[:,2]
        
        # # plot
        # img_dyn = Image.open('/media/ps/passport2/hhy/camelyon16/test/dens_map_sampling_l8/model_l1/save_l3/test_001_heat.png')
        # img_dyn_draw = ImageDraw.ImageDraw(img_dyn)
        # boxes_dyn_show = [[i[0] - i[2] // 2, i[1] - i[2] // 2, i[0] + i[2] // 2, i[1] + i[2] // 2] for i in boxes_nms]
        # for info in boxes_dyn_show:
        #     img_dyn_draw.rectangle(((info[0], info[1]), (info[2], info[3])), fill=None, outline='blue', width=1)
        # img_dyn.save('/media/ps/passport2/hhy/camelyon16/test/heatmap2box_result/crop_split_min_100_l1/dyn_patch.png')
        return POI

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        x_center = int(x_mask * self._resolution)
        y_center = int(y_mask * self._resolution)

        # save_min_100_fix_size_alg_l3
        # patch_size = self._patch_size
        # save_min_100_dyn_size_alg_l3
        patch_size = self._POI[self._X_idcs, self._Y_idcs] * self._resolution
        
        x = int((x_center - patch_size // 2) * self._slide.level_downsamples[self._level_ckpt])
        y = int((y_center - patch_size // 2) * self._slide.level_downsamples[self._level_ckpt])
        
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
        if (right - left) != (r-l):
            a = 1
        return (img, (left, top, right, bot), (l, t, r, b))
        