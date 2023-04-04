import sys
import os
import argparse
import logging
import cv2
import numpy as np
import openslide
from tqdm import tqdm
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('image_path', default=None, metavar='TUMOR_PATH', type=str,
                    help='Path to the output image png file')
parser.add_argument('--level', default=2, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 50')


def run(args):
    logging.basicConfig(level=logging.INFO)
    dir = os.listdir(args.wsi_path)
    for file in tqdm(dir, total=len(dir)):
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file))

        # note the shape of img_RGB is the transpose of slide.level_dimensions
        img_RGB = slide.read_region((0, 0),
                                    args.level,
                                    tuple([int(i / 2**args.level) for i in slide.level_dimensions[0]])).convert('RGB')
        img_RGB = img_RGB.resize(slide.level_dimensions[args.level])
        img_RGB = np.transpose(np.array(img_RGB), axes=[1, 0, 2])
        cv2.imwrite(os.path.join(args.image_path, file.split('.')[0] + '.png'), img_RGB)


def main():
    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/train/tumor",
        "/media/ps/passport2/hhy/camelyon16/train/image_train_l2/tumor"])
    run(args)


if __name__ == '__main__':
    main()
