import os
import json
import glob
import argparse
import openslide
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from camelyon16.cluster.utils import NMM

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
    parser.add_argument('json_dir', default=".", help='the path for box of interest data of json format')
    parser.add_argument('dens_dir', default=".", help='the path for density_map data of json format')
    parser.add_argument('iou_threshold', type=float, help='Threshold defined to select the cropped region')
    parser.add_argument('--label_folder', help='The dir to annotations')
    parser.add_argument('--output_folder', help='The dir to save generated images and annotations')

    args = parser.parse_args(['/media/ps/passport2/hhy/camelyon16/train/crop_split_l3/results.json', 
                              '/media/ps/passport2/hhy/camelyon16/train/dens_map_sliding_l3', '0.3'])
    args.label_folder = '/media/ps/passport2/hhy/camelyon16/train/tumor_mask_l3/'
    args.output_folder = '/media/ps/passport2/hhy/camelyon16/train/crop_cluster_l3'
    return args


if __name__ == "__main__":
    # in data folder, val -> original var data+ val density gt
    # val_mcnn -> mcnn generated data+ mcnn generated density map
    # to work in mcnn, need to copy generated folder to mcnn
    # then run two files. Change root to crop_data_mcnn accordingly
    args = parse_args()
    save_dict = {}
    level = int(args.output_folder.split('l')[-1])
    with open(args.json_dir, 'r') as f2:
        candidates = json.load(f2)
    for file_name in tqdm(candidates.keys(), total=len(candidates.keys())):
        slide = openslide.OpenSlide(file_name)
        bbox_info = candidates[file_name]
        dens_map = np.load(os.path.join(args.dens_dir, os.path.basename(file_name).split('.')[0] + '.npy'))
        # size_scale = (slide.level_dimensions[level][1], slide.level_dimensions[level][0])
        size_scale = tuple([int(i / 2**6) for i in slide.level_dimensions[0]])
        dens_map = cv2.resize(dens_map, size_scale, interpolation=cv2.INTER_CUBIC).transpose() / 255
        scale = 2 ** (level - 6)
        boxes = []
        for box in bbox_info:
            left = int(box[0][0] * scale)
            right = int(box[1][0] * scale)
            top = int(box[0][1] * scale)
            bot = int(box[1][1] * scale)
            area = (right - left) * (bot - top)
            score = dens_map[left:right, top:bot].mean()
            boxes.append([box[0][0], box[0][1], box[1][0], box[1][1], score, box[2]])
        boxes = np.array(boxes)
        cluster_boxes_list, cluster_boxes_dict = NMM(boxes, args.iou_threshold)

        # img show
        level_show = 4
        img = slide.read_region((0, 0), level_show,
                        tuple([int(i / 2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        # img = img.resize(slide.level_dimensions[level_show])
        img_draw = ImageDraw.ImageDraw(img)

        scale_show = 2 ** (level - level_show)
        for cluster in cluster_boxes_dict:
            for child in cluster['child']:
                chi_box = child['cluster_box']
                img_draw.rectangle(((chi_box[0] * scale_show, chi_box[1] * scale_show), \
                    ((chi_box[0]-1+chi_box[2]) * scale_show, (chi_box[1]-1+chi_box[3]) * scale_show)), fill=None, outline='blue', width=5)
            clu_box = cluster['cluster_box']
            img_draw.rectangle(((clu_box[0] * scale_show, clu_box[1] * scale_show), \
                    ((clu_box[0]-1+clu_box[2]) * scale_show, (clu_box[1]-1+clu_box[3]) * scale_show)), fill=None, outline='green', width=5)
        img.save(os.path.join(args.output_folder, os.path.basename(file_name).split('.')[0] + '.png'))

        # add annotation
        # label = np.load(os.path.join(args.label_folder, os.path.basename(file_name).split('.')[0] + '.npy'))
        # for cluster in cluster_boxes_dict:
        #      clu_box = cluster['cluster_box']
        #      density = label[clu_box[0]:clu_box[0]+clu_box[2], clu_box[1]:clu_box[1]+clu_box[3]]
        #      cluster.update({'object_box': density})
        #      for child in cluster['child']:
        #         chi_box = child['cluster_box']
        #         density = label[chi_box[0]:chi_box[0]+chi_box[2], chi_box[1]:chi_box[1]+chi_box[3]]
        #         child.update({'object_box': density})

        save_dict.update({file_name.split('train/')[1]: cluster_boxes_dict})

    # save dict
    with open(os.path.join(args.output_folder, 'results.json'), 'w') as result_file:
        json.dump(save_dict, result_file)
        
