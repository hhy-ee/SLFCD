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
    parser.add_argument('root_dir', default=".",
                        help='the path for source data')
    parser.add_argument('window_size_threshold', help='The size of kernel, format: h_w')
    parser.add_argument('density_prob_threshold', type=float, help='Threshold defined to select the cropped region')
    parser.add_argument('--output_folder', help='The dir to save generated images and annotations')
    args = parser.parse_args(['/media/ps/passport2/hhy/camelyon16/train', '50_50', '0.5'])
    args.output_folder = '/media/ps/passport2/hhy/camelyon16/train/crop_split_l3'
    return args


if __name__ == "__main__":
    # in data folder, val -> original var data+ val density gt
    # val_mcnn -> mcnn generated data+ mcnn generated density map
    # to work in mcnn, need to copy generated folder to mcnn
    # then run two files. Change root to crop_data_mcnn accordingly
    args = parse_args()
    root_dir = args.root_dir
    folder_name = args.output_folder
    level = int(args.output_folder.split('l')[-1])

    img_array = glob.glob(f'{root_dir}/{"tumor"}/*.tif')
    anno_path = glob.glob(f'{root_dir}/{"dens_map_sliding_l{}".format(level)}/*.npy')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=False)
    window_size_threshold = args.window_size_threshold.split("_")
    window_size_threshold = (int(window_size_threshold[0]), int(window_size_threshold[1]))
    save_cropped_result(img_array, window_size_threshold, level, args.density_prob_threshold, args.output_folder)