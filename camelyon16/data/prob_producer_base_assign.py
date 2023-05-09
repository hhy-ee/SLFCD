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
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._canvas = []
        max_box_len = max([len(item['render_seq']) for item in self._assign])
        for assign in self._assign:
            boxes = np.zeros((max_box_len, 4), dtype=np.int)
            moved_boxes = np.zeros((max_box_len, 4), dtype=np.int)
            canvas = np.zeros((assign['bin_height'], assign['bin_width'], 3))
            for idx in assign['render_seq']:
                box = assign['origin_cluster_box'][idx]
                box[2] = max(box[2], box[0] + 1)
                box[3] = max(box[3], box[1] + 1)
                moved_box = assign['moved_cluster_box'][idx]
                o_x1, o_y1, o_x2, o_y2 = box
                w = o_x2 - o_x1
                h = o_y2 - o_y1
                o_s_x1 = int(o_x1 * self._slide.level_downsamples[self._level_ckpt])
                o_s_y1 = int(o_y1 * self._slide.level_downsamples[self._level_ckpt])
                patch = self._slide .read_region((o_s_x1, o_s_y1), self._level_ckpt, (w, h)).convert('RGB')
                patch = np.asarray(patch).transpose((1,0,2))
                x1, y1, x2, y2 = moved_box

                w = x2 - x1
                h = y2 - y1
                int_w = max(1, int(w))
                int_h = max(1, int(h))

                patch = cv2.resize(patch, (int_h, int_w), interpolation=cv2.INTER_CUBIC)
                canvas[int(x1): int(x1) + int_w, int(y1): int(y1) + int_h] = patch

                boxes[idx, :] = box
                moved_boxes[idx, :] = [int(x1), int(y1), int(x1) + int_w, int(y1) + int_h]

            self._canvas.append([canvas, boxes, moved_boxes])
        self._idcs_num = len(self._canvas)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        img, boxes, moved_boxes = self._canvas[idx]

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
        img = img.astype("float32").transpose(2, 0, 1)

        if self._normalize:
            img = (img - 128.0) / 128.0

        return (img, np.array(boxes), np.array(moved_boxes))