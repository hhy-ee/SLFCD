from tqdm import tqdm
import glob
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from camelyon16.cluster.utils import save_cropped_result

"""
Code for DMnet, density crops generation
Author: Changlin Li
Code revised on : 7/16/2020

Given dataset(train/val/test) generate density crops for given dataset.
Default format for source data: The input images are in jpg format and raw annotations are in txt format 
(Based on Visiondrone 2018/19/20 dataset)

The data should be arranged in following structure before you call any function within this script:
dataset(Train/val/test)
--------images
--------dens (short for density map)
--------Annotations (Optional, but not available only when you conduct inference steps)

Sample running command:
python density_slide_window_official.py . height_width threshld --output_folder output_folder --mode val
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='DMNet--Generate density crops from given density map')
    parser.add_argument('wsi_path', default=".", help='the path for source tif data')
    parser.add_argument('densmap_path', default=".", help='the path for density map data')
    parser.add_argument('output_path', default=".", help='the path for save data')
    parser.add_argument('--max_window_size', help='The size of max kernel, format: h_w')
    parser.add_argument('--min_window_size', help='The size of min kernel, format: h_w')
    parser.add_argument('--dens_prob_thres', type=float, help='Threshold defined to select the cropped region')
    

    # args = parser.parse_args(['/media/ps/passport2/hhy/camelyon16/train', '50_50', '0.5'])
    # args.output_folder = '/media/ps/passport2/hhy/camelyon16/train/crop_split_l3'

    args = parser.parse_args(['/media/hy/hhy_data/camelyon16/train/tumor', 
                              '/media/hy/hhy_data/camelyon16/train/dens_map_ncrf_l8',
                              '/media/hy/hhy_data/camelyon16/train/crop_split_ncrf_l0'])
    args.max_window_size = '4096_4096'
    args.min_window_size = '128_128'
    args.dens_prob_thres = 0.1
    return args


if __name__ == "__main__":
    # in data folder, val -> original var data+ val density gt
    # val_mcnn -> mcnn generated data+ mcnn generated density map
    # to work in mcnn, need to copy generated folder to mcnn
    # then run two files. Change root to crop_data_mcnn accordingly
    args = parse_args()
    folder_name = args.output_path
    output_level = int(args.output_path.split('l')[-1])
    densmap_level = int(args.densmap_path.split('l')[-1])
    img_array = glob.glob(f'{args.wsi_path}/*.tif')
    anno_path = glob.glob(f'{args.densmap_path}/*.npy')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=False)
    max_window_size = args.max_window_size.split("_")
    max_window_size = (int(max_window_size[0]), int(max_window_size[1]))
    min_window_size = args.min_window_size.split("_")
    min_window_size = (int(min_window_size[0]), int(min_window_size[1]))
    save_cropped_result(img_array, max_window_size, min_window_size, args.dens_prob_thres, \
                        args.densmap_path, args.output_path, densmap_level, output_level)