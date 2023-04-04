import os
import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
from PIL import ImageDraw

class WSIPatchDataset(Dataset):

    def __init__(self, slide, tissue, level, assign, image_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self.x_scale = slide.level_dimensions[0][0] / 2**level / slide.level_dimensions[level][0]
        self.y_scale = slide.level_dimensions[0][1] / 2**level / slide.level_dimensions[level][1]
        self._image_size = tuple([int(i / 2**level) for i in slide.level_dimensions[0]])
        self._tissue = PIL.Image.fromarray(tissue.transpose()).resize(self._image_size)
        self._level = level
        self._assign = assign
        self._patch_size = image_size
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
                w = o_r - o_l + 1
                h = o_b - o_t + 1
                o_s_l = int(o_l * self.x_scale * self._slide.level_downsamples[self._level])
                o_s_t = int(o_t * self.y_scale * self._slide.level_downsamples[self._level])
                patch = self._slide.read_region((o_s_l, o_s_t), self._level, (w, h)).convert('RGB')

                m_l, m_t, m_r, m_b = map(int, assign['moved_cluster_box'][i])
                m_width, m_height = int(m_r-m_l+1), int(m_b-m_t+1)
                patch_resize = patch.resize((m_width, m_height))
                canvas.paste(patch_resize, (m_l, m_t, m_r+1, m_b+1))

            self._canvas.append([canvas, assign])

        # # plot
        # count = 0
        # for canvas in self._canvas:
        #     canvas[0].save(os.path.join('/media/ps/passport2/hhy/camelyon16/train/crop_assign_l{}'.format(self._level), \
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