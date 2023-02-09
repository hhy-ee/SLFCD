import sys
import os
import argparse
import logging
import time
import numpy as np
from shutil import copyfile
from PIL import Image
from multiprocessing import Pool, Value, Lock

import openslide
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


parser = argparse.ArgumentParser(description='Generate patches from a given '
                                 'list of coordinates')
parser.add_argument('tumor_path', default=None, metavar='IMG_PATH', type=str,
                    help='Path to the input directory of tumor files')
parser.add_argument('coords_path', default=None, metavar='COORDS_PATH',
                    type=str, help='Path to the input list of coordinates')
parser.add_argument('patch_path', default=None, metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=0, type=int, help='level for WSI, to '
                    'generate patches, default 0')
parser.add_argument('--num_process', default=1, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(opts):
    i, pid, x_center, y_center, label, args = opts
    x = int(int(x_center) - args.patch_size / 2)
    y = int(int(y_center) - args.patch_size / 2)
    wsi_img = Image.open(os.path.join(os.path.join(args.tumor_path, 'wsi_image'), pid + '.png'))
    
    img_patch = wsi_img.crop((x, y, x + args.patch_size, y + args.patch_size))
    img_patch.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '.png'))

    # tumor_mask = np.load(os.path.join(args.tumor_path, pid + '.npy'))
    # segmentation = np.zeros((args.patch_size, args.patch_size)) > 127
    # w, h = tumor_mask.shape
    # segmentation[: w - x, : h - y] = 
    # np.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '.npy'), segmentation)

    tumor_mask = Image.fromarray(np.load(os.path.join(args.tumor_path, pid + '.npy')).transpose())
    segmentation = tumor_mask.crop((x, y, x + args.patch_size, y + args.patch_size))
    segmentation.save(os.path.join(os.path.join(args.patch_path, pid), str(i) + '_seg.png'))
    
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
    for file in dir:
        if file.split('.')[-1] == 'txt':
            if not os.path.exists(os.path.join(args.patch_path, file.split('.')[0])):
                os.mkdir(os.path.join(args.patch_path, file.split('.')[0]))

            copyfile(os.path.join(args.coords_path, file.split('.')[0] + '.txt') , \
                os.path.join(os.path.join(args.patch_path, file.split('.')[0]), 'list.txt'))

            opts_list = []
            infile = open(os.path.join(args.coords_path, file.split('.')[0] + '.txt'))
            for i, line in enumerate(infile):
                pid, x_center, y_center, label = line.strip('\n').split(',')
                opts_list.append((i, pid, x_center, y_center, label, args))
            infile.close()

            pool = Pool(processes=args.num_process)
            pool.map(process, opts_list)


def main():
    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/training/tumor_mask_l6",
        "/media/ps/passport2/hhy/camelyon16/training/sample_gen/",
        "/media/ps/passport2/hhy/camelyon16/training/patch_gen_baseline/"])
    args.patch_size = 128
    args.level = 6
    run(args)

if __name__ == '__main__':
    main()