import os
import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
from PIL import ImageDraw

class WSIPatchDataset(Dataset):

    def __init__(self, slide, assign, level, overlap, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._assign = assign
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
        self._canvas = []
        for assign in self._assign:
            canvas = PIL.Image.new('RGB', (assign['bin_width'], assign['bin_height']), "white")
            for i in range(len(assign['cluster_box'])):
                o_l, o_t, o_r, o_b = map(int, assign['origin_cluster_box'][i])
                patch = self._img.crop((o_l, o_t, o_r+1, o_b+1))
                m_l, m_t, m_r, m_b = map(int, assign['moved_cluster_box'][i])
                m_width, m_height = int(m_r-m_l+1), int(m_b-m_t+1)
                patch_resize = patch.resize((m_width, m_height))
                canvas.paste(patch_resize, (m_l, m_t, m_r+1, m_b+1))

            self._canvas.append([canvas, assign])

        # # plot
        # count = 0
        # for canvas in self._canvas:
        #     canvas[0].save(os.path.join('/media/ps/passport2/hhy/camelyon16/train/crop_assign', \
        #         assign['file_name'].split('/')[-1].split('.')[0] + '_{}'.format(count) + '.png'))
        #     count += 1

        self._idcs_num = len(self._canvas)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        img, assign = self._canvas[idx]

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

        return (img, assign)

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