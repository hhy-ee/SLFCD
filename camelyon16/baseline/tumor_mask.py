import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the JSON file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--vis_result', default=False, type=bool, help='whether'
                    ' to show the results')


def run(args):
    dir = os.listdir(args.json_path)
    for file in tqdm(dir, total=len(dir)):
        # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0] + '.tif'))
        level = int(args.npy_path.split('l')[-1])
        w, h = slide.level_dimensions[level]
        mask_tumor = np.zeros((h, w)) # the init mask, and all the value is 0
        if 'tumor' in file or 'test' in file:
            # get the factor of level * e.g. level 6 is 2^6
            factor = (slide.level_dimensions[0][0]/w, slide.level_dimensions[0][1]/h)

            with open(os.path.join(args.json_path, file)) as f:
                dicts = json.load(f)
            tumor_polygons = dicts['positive']

            for tumor_polygon in tumor_polygons:
                # plot a polygon
                name = tumor_polygon["name"]
                vertices = np.array(tumor_polygon["vertices"]) / factor
                vertices = vertices.astype(np.int32)

                cv2.fillPoly(mask_tumor, [vertices], (255))

            # mask_tumor = mask_tumor[:] > 127
            # mask_tumor = np.transpose(mask_tumor)
            # np.save(os.path.join(args.npy_path, file.split('.')[0] + '.npy'), mask_tumor)

            size = tuple([int(i / 2**level) for i in slide.level_dimensions[0]])
            mask_tumor = cv2.resize(mask_tumor.astype(np.uint8), size, interpolation=cv2.INTER_CUBIC)
            mask_tumor = np.transpose(mask_tumor)
            mask_tumor = mask_tumor[:] > 127
            np.save(os.path.join(args.npy_path, file.split('.')[0] + '.npy'), mask_tumor)

            if args.vis_result:
                img_tumor = slide.read_region((0, 0), level,
                                    tuple([int(i / 2**level) for i in slide.level_dimensions[0]])).convert('RGB')
                mask_tumor = cv2.applyColorMap((mask_tumor * 255).astype(np.uint8), cv2.COLORMAP_JET)
                mask_tumor = Image.fromarray(cv2.cvtColor(mask_tumor, cv2.COLOR_BGR2RGB).transpose((1,0,2)))
                heat_img = Image.blend(img_tumor, mask_tumor, 0.3)
                heat_img.save(os.path.join(os.path.join(args.npy_path, 'result'), file.split('.')[0] + '.png'))

def main():
    logging.basicConfig(level=logging.INFO)

    # args = parser.parse_args([
    #     "/media/ps/passport2/hhy/camelyon16/train/tumor",
    #     "/media/ps/passport2/hhy/camelyon16/train/annotations/json",
    #     "/media/ps/passport2/hhy/camelyon16/train/tumor_mask_l5"])
    # args.vis_result = False

    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/test/images",
        "/media/ps/passport2/hhy/camelyon16/test/annotations/json",
        "/media/ps/passport2/hhy/camelyon16/test/tumor_mask_l5"])
    args.vis_result = False
    run(args)

if __name__ == "__main__":
    main()
