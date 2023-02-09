import os
import sys

import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image

np.random.seed(0)

from torchvision import transforms  # noqa


class ImageDataset(Dataset):

    def __init__(self, data_path, img_size,
                 crop_size=224, normalize=True):
        self._data_path = data_path
        self._img_size = img_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._pre_process()

    def _pre_process(self):
        # find classes
        if sys.version_info >= (3, 5):
            # Faster and available in python 3.5 and above
            classes = [d.name for d in os.scandir(self._data_path) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._data_path) if os.path.isdir(os.path.join(self._data_path, d))]
        classes.sort()

        # make dataset
        self._items = []
        for target in sorted(classes):
            d = os.path.join(self._data_path, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.split('_')[-1] == 'seg.png':
                        label_path = os.path.join(root, fname)
                        img_path = os.path.join(root, fname.split('_')[0] + '.png')
                        item = (img_path, label_path)
                        self._items.append(item)

        random.shuffle(self._items)

        self._num_images = len(self._items)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        img_path, label_path = self._items[idx]

        label = Image.open(label_path)

        img = Image.open(img_path)

        # color jitter
        img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)
        label = label.rotate(90 * num_rotate)

        # PIL image: H W C
        # torch image: C H W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
        label = np.array(label, dtype=np.float32)

        if self._normalize:
            img = (img - 128.0) / 128.0

        return img, label