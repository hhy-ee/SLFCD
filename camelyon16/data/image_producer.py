import os
import sys

import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image

np.random.seed(0)

from torchvision import transforms  # noqa
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

class ImageDataset(Dataset):

    def __init__(self, data_path, normalize=True):
        self._data_path = data_path
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._pre_process()

    def _pre_process(self):
        # find classes
        # Faster and available in python 3.5 and above
        classes = [d.name for d in os.scandir(self._data_path) if d.is_dir()]
        # make dataset
        self.total_items = []
        for target in sorted(classes):
            d = os.path.join(self._data_path, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if 'img' in fname:
                        img_path = os.path.join(root, fname)
                        label_path = os.path.join(root, fname.split('_')[0] + '_label.png')
                        item = (img_path, label_path)
                        self.total_items.append(item)
        
        random.shuffle(self.total_items)
        self._num_images = len(self.total_items)

    def __len__(self):
        return len(self.local_items)

    def data_split(self, mode):
        if mode == 'train':
            self.local_items = self.total_items[self._num_images // 10:]
        elif mode == 'valid':
            self.local_items = self.total_items[:self._num_images // 10]
            
    def data_shuffle(self):
        random.shuffle(self.local_items)
        
    def __getitem__(self, idx):
        img_path, label_path = self.local_items[idx]

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