import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
import scipy.signal

class WSIPatchDataset(Dataset):

    def __init__(self, slide, level, subdivisions, image_size,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._slide = slide
        self._level = level
        self._subdivisions = subdivisions
        self._image_size = image_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._img = self._slide.read_region((0, 0), self._level,
                        tuple([int(i / 2**self._level) for i in self._slide.level_dimensions[0]])).convert('RGB')
        self._img = self._img.resize(self._slide.level_dimensions[self._level])

        if self._subdivisions != 0:
            self._aug = int(round(self._image_size * (1 - 1.0/self._subdivisions)))
            more_borders = ((self._aug, self._aug), (self._aug, self._aug), (0, 0))
            img = np.array(self._img, dtype=np.float32).transpose((1, 0, 2))
            if self._normalize:
                img = (img - 128.0) / 128.0
            self._pad_img = np.pad(img, pad_width=more_borders, mode='reflect').transpose((2, 0, 1))

            self.WINDOW_SPLINE_2D = self._window_2D(window_size=self._image_size, power=2)
            step = int(self._image_size/self._subdivisions)
            self.pad_x_len = self._pad_img.shape[1]
            self.pad_y_len = self._pad_img.shape[2]

            X_mesh_idcs = np.arange(0, self.pad_x_len-self._image_size+1, step)
            Y_mesh_idcs = np.arange(0, self.pad_y_len-self._image_size+1, step)

            X_idcs_mesh, Y_idcs_mesh = np.meshgrid(X_mesh_idcs, Y_mesh_idcs)
            self._X_idcs = X_idcs_mesh.flatten()
            self._Y_idcs = Y_idcs_mesh.flatten()

            self._idcs_num = len(self._X_idcs)
        else:
            X_slide, Y_slide = self._slide.level_dimensions[self._level]
            
            X_idcs_num = X_slide // self._image_size
            Y_idcs_num = Y_slide // self._image_size
            
            X_mesh_idcs = np.arange(X_idcs_num) * self._image_size
            Y_mesh_idcs = np.arange(Y_idcs_num) * self._image_size

            if X_slide % self._image_size != 0:
                X_mesh_idcs = np.arange(X_idcs_num + 1) * self._image_size
            if Y_slide % self._image_size != 0:
                Y_mesh_idcs = np.arange(Y_idcs_num + 1) * self._image_size

            X_idcs_mesh, Y_idcs_mesh = np.meshgrid(X_mesh_idcs, Y_mesh_idcs)
            self._X_idcs = X_idcs_mesh.flatten()
            self._Y_idcs = Y_idcs_mesh.flatten()

            self._idcs_num = len(self._X_idcs)

    def _window_2D(self, window_size, power=2):
        """
        Make a 1D window function, then infer and return a 2D window function.
        Done with an augmentation, and self multiplication with its transpose.
        Could be generalized to more dimensions.
        """
        wind = self._spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 2)
        wind = wind * wind.transpose(1, 0, 2)

        return wind

    def _spline_window(self, window_size, power=2):
        """
        Squared spline (power=2) window function:
        """
        intersection = int(window_size/4)
        wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x, y = self._X_idcs[idx], self._Y_idcs[idx]
        img = self._pad_img[:, x:x + self._image_size, y:y + self._image_size]

        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self._rotate == 'ROTATE_90':
            img = img.transpose(PIL.Image.ROTATE_90)

        if self._rotate == 'ROTATE_180':
            img = img.transpose(PIL.Image.ROTATE_180)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(PIL.Image.ROTATE_270)

        return (img, (x, y, x + self._image_size, y + self._image_size), (self._image_size, self._image_size))

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