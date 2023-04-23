import sys
import os
import argparse
import logging
import time
import json
import cv2
import openslide
import numpy as np

from tqdm import tqdm
from shutil import copyfile
from PIL import Image
from multiprocessing import Pool, Value, Lock

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


parser = argparse.ArgumentParser(description='Generate patches from a given '
                                 'list of coordinates')
parser.add_argument('wsi_path', default=None, metavar='IMG_PATH', type=str,
                    help='Path to the input directory of tif files')
parser.add_argument('coords_path', default=None, metavar='COORDS_PATH',
                    type=str, help='Path to the input list of coordinates')
parser.add_argument('patch_path', default=None, metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--num_process', default=1, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(opts, slide, level, tumor_mask=None):
    i, pid, x_center, y_center, args = opts

    x_scale = int(x_center) * slide.level_dimensions[0][0] / 2**level / slide.level_dimensions[level][0]
    y_scale = int(y_center) * slide.level_dimensions[0][1] / 2**level / slide.level_dimensions[level][1]

    x = int(x_scale - args.patch_size / 2)
    y = int(y_scale - args.patch_size / 2)

    x_crop = int(x * slide.level_downsamples[level])
    y_crop = int(y * slide.level_downsamples[level])

    # generate wsi image patch
    img_patch = slide.read_region((x_crop, y_crop), level,
                                    (args.patch_size, args.patch_size)).convert('RGB')
    img_patch.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '_img.png'))

    # generate tumor label
    if 'tumor' in pid:
        mask_tumor = tumor_mask.crop((x, y, x + args.patch_size, y + args.patch_size))
        mask_tumor.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '_label.png'))
    else:
        mask_tumor = Image.fromarray(np.zeros((args.patch_size, args.patch_size)) > 127)
        mask_tumor.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '_label.png'))

    # # generate heat map
    # mask_tumor = (np.asarray(mask_tumor) * 255).astype(np.uint8)
    # mask_tumor = cv2.applyColorMap(mask_tumor, cv2.COLORMAP_JET)
    # mask_tumor = Image.fromarray(cv2.cvtColor(mask_tumor, cv2.COLOR_BGR2RGB))
    # heat_img = Image.blend(img_patch, mask_tumor, 0.3)
    # heat_img.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '_heatmap.png'))

    global lock
    global count

    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))


def run(args):
    logging.basicConfig(level=logging.INFO)
    dir = os.listdir(args.coords_path)
    for file in tqdm(dir, total=len(dir)):
        if 'tumor' in file:
            if not os.path.exists(os.path.join(args.patch_path, file.split('.')[0])):
                os.mkdir(os.path.join(args.patch_path, file.split('.')[0]))

                copyfile(os.path.join(args.coords_path, file.split('.')[0] + '.txt') , \
                    os.path.join(os.path.join(args.patch_path, file.split('.')[0]), 'list.txt'))

                opts_list = []
                infile = open(os.path.join(args.coords_path, file.split('.')[0] + '.txt'))
                for i, line in enumerate(infile):
                    pid, x_center, y_center = line.strip('\n').split(',')
                    opts_list.append((i, pid, x_center, y_center, args))
                infile.close()

                wsi_path = os.path.join(args.wsi_path, pid.split('_')[0], pid + '.tif')
                slide = openslide.OpenSlide(wsi_path)
                level = int(args.patch_path.split('l')[-1]) # 0

                if 'tumor' in file:
                    tumor_mask = np.load(os.path.join(args.wsi_path, 'tumor_mask_l1', pid + '.npy')).transpose()
                    tumor_mask_img = Image.fromarray(tumor_mask).resize(tuple([int(i / 2**level) for i in slide.level_dimensions[0]]))
                    tumor_mask_img_full = Image.new(tumor_mask_img.mode, (slide.level_dimensions[level]))
                    tumor_mask_img_full.paste(tumor_mask_img, (0, 0)+tuple([int(i / 2**level) for i in slide.level_dimensions[0]]))
                    for opts in opts_list:
                        process(opts, slide, level, tumor_mask_img_full)
                else:
                    for opts in opts_list:
                        process(opts, slide, level)


def main():
    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/train",
        "/media/ps/passport2/hhy/camelyon16/train/sample_gen_l0",
        "/media/ps/passport2/hhy/camelyon16/train/patch_gen_l0"])
    args.patch_size = 256

    run(args)

if __name__ == '__main__':
    main()