import os
import json
import glob
import argparse
import openslide
import sys
import cv2
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage as nd
from skimage import measure
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
    parser.add_argument('wsi_path', default=".", help='the path for source tif data')
    parser.add_argument('densmap_path', default=".", help='the path for density map data')
    parser.add_argument('output_path', default=".", help='the path for save data')
    parser.add_argument('--ITC_threshold', help='The size of min kernel, format: h_w')
    parser.add_argument('--roi_generator', default='base_l8', metavar='ROI_GENERATOR',
                    type=str, help='type of the generator of the first stage')
    parser.add_argument('--roi_threshold', default=0.1, metavar='ROI_GENERATOR',
                    type=float, help='type of the generator of the first stage')

    # args = parser.parse_args(['/media/ps/passport2/hhy/camelyon16/train', '50_50', '0.5'])
    # args.output_folder = '/media/ps/passport2/hhy/camelyon16/train/crop_split_l3'

    # args = parser.parse_args(['/media/hy/hhy_data/camelyon16/test/images', 
    #                           '/media/hy/hhy_data/camelyon16/test/dens_map_sampling_l8',
    #                           '/media/hy/hhy_data/camelyon16/test/crop_split_2048_sampling_l1'])
    # args.max_window_size = '2048_2048'
    # args.min_window_size = '128_128'
    # args.dens_prob_thres = 0.1

    args = parser.parse_args(['/media/ps/passport2/hhy/camelyon16/test/images', 
                              '/media/ps/passport2/hhy/camelyon16/test/dens_map_sampling_l8',
                              '/media/ps/passport2/hhy/camelyon16/test/heatmap2box_result/crop_split_min_200_l1'])
    args.ITC_threshold = 200    # ITC_threshold / (0.243 * pow(2, level))
    args.roi_generator = 'sampling_l8'
    args.roi_threshold = 0.1

    return args

if __name__ == "__main__":

    args = parse_args()
    save_dict = {}
    time_total = 0.0
    
    level_show = 6
    # dens_level = int(args.dens_path.split('l')[-1])
    level_dens = 6
    level_outp = int(args.output_path.split('l')[-1])

    dir = os.listdir(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l{}'.format(level_dens)))
    
    for file in tqdm(sorted(dir), total=len(dir)):
        # if os.path.exists(os.path.join(args.output_path, os.path.basename(file_name).replace('.tif','.png'))):
        #     continue

        # calculate score of each patches
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        tissue = np.load(os.path.join(os.path.dirname(args.densmap_path), 'dens_map_{}'.format(args.roi_generator), 'model_l1', 'save_l3', file))
        tissue_shape = tuple([int(i / 2**level_dens) for i in slide.level_dimensions[0]])
        tissue = cv2.resize(tissue, (tissue_shape[1], tissue_shape[0]), interpolation=cv2.INTER_CUBIC)
        POI = (tissue / 255) > args.roi_threshold
        
        # Computes the inference mask
        filled_image = nd.morphology.binary_fill_holes(POI)
        evaluation_mask = measure.label(filled_image, connectivity=2)
        
        # eliminate ITC
        max_label = np.amax(evaluation_mask)
        properties = measure.regionprops(evaluation_mask)
        filled_mask = np.zeros(tissue.shape) > 0
        threshold = args.ITC_threshold / (0.243 * pow(2, level_dens))
        for i in range(0, max_label):
            if properties[i].major_axis_length > threshold:
                l, t, r, b = properties[i].bbox
                filled_mask[l: r, t: b] = np.logical_or(filled_mask[l: r, t: b], properties[i].image_filled)
        tissue = tissue * filled_mask
        
        # plot
        img_rgb = slide.read_region((0, 0), level_show, \
                    tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        tissue_rgb = cv2.applyColorMap(tissue.astype(np.uint8), cv2.COLORMAP_JET)
        tissue_rgb = cv2.cvtColor(tissue_rgb, cv2.COLOR_BGR2RGB)
        heat_img = cv2.addWeighted(tissue_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        img = Image.fromarray(heat_img)
        img_draw = ImageDraw.ImageDraw(img)
        for i in range(0, max_label):
            if properties[i].major_axis_length > threshold:
                l, t, r, b  = properties[i].bbox
                img_draw.rectangle(((l, t), (r, b)), fill=None, outline='blue', width=1)
        heat_img_rect = np.asarray(img)
        cv2.imwrite(os.path.join(args.output_path, file.split('.')[0] + '_heat.png'), heat_img_rect)
        
        # distance map
        POI = (tissue / 255) > args.roi_threshold
        distance, coord = nd.distance_transform_edt(POI, return_indices=True)
        np.save(os.path.join(args.output_path, file), distance)
        
