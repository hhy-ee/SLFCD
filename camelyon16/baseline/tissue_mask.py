import sys
import os
import argparse
import logging

import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 50')


def run(args):
    logging.basicConfig(level=logging.INFO)
    dir = os.listdir(args.wsi_path)
    for file in dir:
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file))

        # note the shape of img_RGB is the transpose of slide.level_dimensions
        img_RGB = slide.read_region((0, 0),
                                    args.level,
                                    tuple([int(i / 2**args.level) for i in slide.level_dimensions[0]])).convert('RGB')
        img_RGB = img_RGB.resize(slide.level_dimensions[args.level])
        img_RGB = np.transpose(np.array(img_RGB), axes=[1, 0, 2])

        img_HSV = rgb2hsv(img_RGB)

        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
        min_R = img_RGB[:, :, 0] > args.RGB_min
        min_G = img_RGB[:, :, 1] > args.RGB_min
        min_B = img_RGB[:, :, 2] > args.RGB_min

        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

        np.save(os.path.join(args.npy_path, file.split('.')[0] + '.npy'), tissue_mask)

def main():
    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/training/tumor",
        "/media/ps/passport2/hhy/camelyon16/training/tissue_mask_l6"])
    run(args)


if __name__ == '__main__':
    main()
