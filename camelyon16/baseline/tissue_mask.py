import sys
import os
import argparse
import logging

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
parser.add_argument('tissue_path', default=None, metavar='TISSUE_PATH', type=str,
                    help='Path to the output tissue npy mask file')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 50')


def run(args):
    logging.basicConfig(level=logging.INFO)
    dir = os.listdir(args.wsi_path)
    for file in tqdm(sorted(dir), total=len(dir)):
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file))
        level = int(args.tissue_path.split('l')[-1])
        # note the shape of img_RGB is the transpose of slide.level_dimensions
        img_RGB = slide.read_region((0, 0), level,
                                    tuple([int(i / 2**level) for i in slide.level_dimensions[0]])).convert('RGB')
        # img_RGB = img_RGB.resize(slide.level_dimensions[level])
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
        if 'tumor' in file or 'test' in file:
            tumor_path = os.path.join(os.path.dirname(args.tissue_path), 'tumor_mask_l{}'.format(level))
            if os.path.exists(os.path.join(tumor_path, file.split('.')[0] + '.npy')):
                tumor_mask = np.load(os.path.join(tumor_path, file.split('.')[0] + '.npy'))
                tissue_mask = tissue_mask | tumor_mask
        np.save(os.path.join(args.tissue_path, file.split('.')[0] + '.npy'), tissue_mask)

def main():
    # args = parser.parse_args([
    #     "/media/ps/passport2/hhy/camelyon16/train/tumor",
    #     "/media/ps/passport2/hhy/camelyon16/train/tissue_mask_l0"])
    
    # args = parser.parse_args([
    #     "/media/ps/passport2/hhy/camelyon16/test/images",
    #     "/media/ps/passport2/hhy/camelyon16/test/tissue_mask_l5"])
    
    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/train/tumor",
        "/media/ps/passport2/hhy/camelyon16/train/tissue_mask_l5"])
    
    run(args)


if __name__ == '__main__':
    main()
