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
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the JSON file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')


def run(args):
    dir = os.listdir(args.wsi_path)
    for file in dir:
        # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file))
        w, h = slide.level_dimensions[args.level]
        mask_tumor = np.zeros((h, w)) # the init mask, and all the value is 0
        if 'tumor' in file:
            # get the factor of level * e.g. level 6 is 2^6
            factor = (slide.level_dimensions[0][0]/w, slide.level_dimensions[0][1]/h)

            with open(os.path.join(args.json_path, file.split('.')[0] + '.json')) as f:
                dicts = json.load(f)
            tumor_polygons = dicts['positive']

            for tumor_polygon in tumor_polygons:
                # plot a polygon
                name = tumor_polygon["name"]
                vertices = np.array(tumor_polygon["vertices"]) / factor
                vertices = vertices.astype(np.int32)

                cv2.fillPoly(mask_tumor, [vertices], (255))
            
        wsi_img = slide.read_region((0, 0),
                                args.level,
                                tuple([int(i / 2**args.level) for i in slide.level_dimensions[0]])).convert('RGB')
        wsi_img = wsi_img.resize(slide.level_dimensions[args.level])
        wsi_img.save(os.path.join(os.path.join(args.npy_path, 'wsi_image'), file.split('.')[0] + '.png'))

        mask_tumor = mask_tumor[:] > 127
        mask_tumor = np.transpose(mask_tumor)
        np.save(os.path.join(args.npy_path, file.split('.')[0] + '.npy'), mask_tumor)

        mask_img = np.asarray(mask_tumor * 255, dtype=np.uint8)
        mask_img = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
        mask_img = Image.fromarray(mask_img.transpose((1, 0, 2)))
        heat_img = Image.blend(wsi_img, mask_img, 0.3)
        heat_img.save(os.path.join(os.path.join(args.npy_path, 'heat_image'), file.split('.')[0] + '.png'))

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/training/tumor",
        "/media/ps/passport2/hhy/camelyon16/training/annotations/json",
        "/media/ps/passport2/hhy/camelyon16/training/tumor_mask_l61"])
    run(args)

if __name__ == "__main__":
    main()
