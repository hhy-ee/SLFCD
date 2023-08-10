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
parser.add_argument('--patch_size', default=256, help='patch size, default 256')
parser.add_argument('--patch_mode', default='fix', help='patch size, default 256')
parser.add_argument('--num_process', default=1, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(opts, slide, level, tumor_mask=None):
    i, pid, x_center, y_center, args = opts
    if args.patch_mode == 'fix':
        patch_size = args.patch_size
    elif args.patch_mode == 'dyn':
        patch_size = np.random.randint(2,16) * 32
        
    l = int(x_center) - patch_size // 2
    t = int(y_center) - patch_size // 2
    
    w, h = tuple([int(i / 2**level) for i in slide.level_dimensions[0]])
    l = min(max(0, l), w - patch_size)
    t = min(max(0, t), h - patch_size)
    
    x = int(l * slide.level_downsamples[level])
    y = int(t * slide.level_downsamples[level])

    # generate wsi image patch
    img_patch = slide.read_region((x, y), level, (patch_size,)*2).convert('RGB')
    if args.patch_mode == 'dyn':
        img_patch = img_patch.resize((args.patch_size,)*2)
    img_patch.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '_img.png'))

    # generate tumor label
    if 'tumor' in pid:
        mask_tumor = tumor_mask[l: l + patch_size, t: t + patch_size]
        mask_tumor = Image.fromarray(mask_tumor.transpose())
        if args.patch_mode == 'dyn':
            mask_tumor = mask_tumor.resize((args.patch_size,)*2)
        mask_tumor.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '_label.png'))
    else:
        mask_tumor = Image.fromarray(np.zeros((patch_size, patch_size)) > 127)
        if args.patch_mode == 'dyn':
            mask_tumor = mask_tumor.resize((args.patch_size,)*2)
        mask_tumor.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '_label.png'))

    # # generate heat map
    mask_tumor = (np.asarray(mask_tumor) * 255).astype(np.uint8)
    mask_tumor = Image.fromarray(cv2.applyColorMap(mask_tumor, cv2.COLORMAP_JET))
    heat_img = Image.blend(img_patch, mask_tumor, 0.3)
    heat_img.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '_heat.png'))

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
    for file in tqdm(sorted(dir), total=len(dir)):
        if not 'tumor' in file:
            continue
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
            level = int(args.patch_path.split('l')[-1])

            if 'tumor' in file:
                tumor_mask = np.load(os.path.join(args.wsi_path, 'tumor_mask_l1', pid + '.npy'))
                for opts in opts_list:
                    process(opts, slide, level, tumor_mask)
            else:
                for opts in opts_list:
                    process(opts, slide, level)


def main():
    # args = parser.parse_args([
    #     "./datasets/train",
    #     "./datasets/train/sample_gen_l1",
    #     "./datasets/train/patch_gen_fix_l1"])
    # args.patch_size = 256
    # args.patch_mode = 'fix'
    # run(args)
    
    args = parser.parse_args([
        "./datasets/train",
        "./datasets/train/sample_gen_l0",
        "./datasets/train/patch_gen_fix_l0"])
    args.patch_size = 256
    args.patch_mode = 'fix'
    run(args)

if __name__ == '__main__':
    main()