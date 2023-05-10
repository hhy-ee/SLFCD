import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('png_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output png file')


def run(args):
    dir = os.listdir(args.wsi_path)
    for file in sorted(dir):
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file))
        level = int(args.png_path.split('l')[-1])
        wsi_img = slide.read_region((0, 0), level,
                                tuple([int(i / 2**level) for i in slide.level_dimensions[0]])).convert('RGB')
        wsi_img = wsi_img.resize(slide.level_dimensions[level])
        # wsi_img.save(os.path.join(args.png_path, file.split('.')[0] + '.png'))
        
        # # heatmap of tissue and tumor
        # mask_tissue = np.load(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask', file.split('.')[0] + '.npy'))
        mask_tumor = np.load(os.path.join(os.path.dirname(args.wsi_path), 'tumor_mask_l6', file.split('.')[0] + '.npy'))

        # if not os.path.exists(os.path.join(args.png_path, 'heat_tissue')):
        #         os.mkdir(os.path.join(args.png_path, 'heat_tissue'))
        # if not os.path.exists(os.path.join(args.png_path, 'heat_tumor')):
        #         os.mkdir(os.path.join(args.png_path, 'heat_tumor'))

        # mask_tissue = np.asarray(mask_tissue * 255, dtype=np.uint8)
        # mask_tissue = cv2.applyColorMap(mask_tissue, cv2.COLORMAP_JET)
        # mask_tissue = cv2.cvtColor(mask_tissue, cv2.COLOR_BGR2RGB)
        # mask_tissue = Image.fromarray(mask_tissue.transpose((1, 0, 2))).resize(slide.level_dimensions[args.level])
        # heat_tissue = Image.blend(wsi_img, mask_tissue, 0.5)
        # heat_tissue.save(os.path.join(args.png_path, 'heat_tissue', file.split('.')[0] + '.png'))

        mask_tumor = np.asarray(mask_tumor * 255, dtype=np.uint8)
        mask_tumor = cv2.applyColorMap(mask_tumor, cv2.COLORMAP_JET)
        mask_tumor = cv2.cvtColor(mask_tumor, cv2.COLOR_BGR2RGB)
        mask_tumor = Image.fromarray(mask_tumor.transpose((1, 0, 2))).resize(slide.level_dimensions[level])
        heat_tumor = Image.blend(wsi_img, mask_tumor, 0.5)
        heat_tumor.save(os.path.join(args.png_path, file.split('.')[0] + '.png'))

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/train/tumor",
        "/media/ps/passport2/hhy/camelyon16/train/wsi_image_l3"])
    run(args)

if __name__ == "__main__":
    main()
