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
from camelyon16.cluster.utils import NMS

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
    parser.add_argument('prior_path', default=".", help='the path for density map data')
    parser.add_argument('output_path', default=".", help='the path for save data')
    parser.add_argument('--roi_generator', default='base_l8', metavar='ROI_GENERATOR',
                    type=str, help='type of the generator of the first stage')
    parser.add_argument('--roi_threshold', default=0.1, metavar='ROI_GENERATOR',
                    type=float, help='type of the generator of the first stage')
    parser.add_argument('--itc_threshold', help='The threshold of ITC to be eliminated')
    parser.add_argument('--ini_patchsize', help='The size of initial patch size')
    parser.add_argument('--min_threshold', help='The threshold for patch to stop zooming out')
    parser.add_argument('--nmm_threshold', help='The threshold to select the cropped region')
    
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
                              '/media/ps/passport2/hhy/camelyon16/test/heatmap2box_result/crop_split_min_100_l1'])
    args.roi_generator = 'sampling_l8'
    args.roi_threshold = 0.1
    args.itc_threshold = 100    # ITC_threshold / (0.243 * pow(2, level))
    args.ini_patchsize = 256
    args.min_threshold = 8
    args.nmm_threshold = 0.1
    return args

if __name__ == "__main__":

    args = parse_args()
    save_dict = {}
    time_total = 0.0
    
    level_show = 6
    level_prior = 6
    level_output = int(args.output_path.split('l')[-1])

    dir = os.listdir(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l{}'.format(level_prior)))
    
    for file in tqdm(sorted(dir), total=len(dir)):
        # if os.path.exists(os.path.join(args.output_path, os.path.basename(file_name).replace('.tif','.png'))):
        #     continue

        # calculate score of each patches
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        first_stage_map = np.load(os.path.join(os.path.dirname(args.prior_path), \
                                    'dens_map_{}'.format(args.roi_generator), 'model_l1', 'save_l3', file))
        prior_shape = tuple([int(i / 2**level_prior) for i in slide.level_dimensions[0]])
        first_stage_map = cv2.resize(first_stage_map, \
                                    (prior_shape[1], prior_shape[0]), interpolation=cv2.INTER_CUBIC)
        POI = (first_stage_map / 255) > args.roi_threshold
        
        # Computes the inference mask
        filled_image = nd.morphology.binary_fill_holes(POI)
        evaluation_mask = measure.label(filled_image, connectivity=2)
        
        # eliminate ITC
        max_label = np.amax(evaluation_mask)
        properties = measure.regionprops(evaluation_mask)
        filled_mask = np.zeros(first_stage_map.shape) > 0
        threshold = args.itc_threshold / (0.243 * pow(2, level_prior))
        for i in range(0, max_label):
            if properties[i].major_axis_length > threshold:
                l, t, r, b = properties[i].bbox
                filled_mask[l: r, t: b] = np.logical_or(filled_mask[l: r, t: b], properties[i].image_filled)
        
        # plot
        # img_rgb = slide.read_region((0, 0), level_show, \
        #             tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        # img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        # tissue_rgb = cv2.applyColorMap(tissue.astype(np.uint8), cv2.COLORMAP_JET)
        # tissue_rgb = cv2.cvtColor(tissue_rgb, cv2.COLOR_BGR2RGB)
        # heat_img = cv2.addWeighted(tissue_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        # img = Image.fromarray(heat_img)
        # img_draw = ImageDraw.ImageDraw(img)
        # for i in range(0, max_label):
        #     if properties[i].major_axis_length > threshold:
        #         l, t, r, b  = properties[i].bbox
        #         img_draw.rectangle(((l, t), (r, b)), fill=None, outline='blue', width=1)
        # heat_img_rect = np.asarray(img)
        # cv2.imwrite(os.path.join(args.output_path, file.split('.')[0] + '_heat.png'), heat_img_rect)
        
        # distance map
        distance, coord = nd.distance_transform_edt(filled_mask, return_indices=True)
        edge_X, edege_Y = np.where(distance == 1)
        edge_map = (distance == 1)
        boxes = []
        for idx in range(0, len(edge_X)):
            x_center, y_center = edge_X[idx], edege_Y[idx]
            ini_size = args.ini_patchsize // 2 ** (level_prior - level_output)
            l, t = x_center - ini_size // 2, y_center - ini_size // 2
            while edge_map[l: l+ini_size, t: t+ini_size].sum() > 8:
                ini_size = ini_size - 1
                l, t = x_center - ini_size // 2, y_center - ini_size // 2
            scr = first_stage_map[x_center, y_center]
            boxes.append([l, t, l+ini_size, t+ini_size, scr, x_center, y_center])
            
        scale_save = 2 ** (level_prior - level_output)
        boxes_save = [[int(i[0] * scale_save), int(i[1] * scale_save), int(i[2] * scale_save), \
                        int(i[3] * scale_save), i[4], int(i[5] * scale_save), int(i[6] * scale_save)] for i in boxes]
        
        # NMS
        if len(boxes) == 0:
            continue
        boxes_save = np.array(boxes_save)
        keep_boxes_list, cluster_boxes_dict = NMS(boxes_save, args.nmm_threshold)
        boxes_save = [list(i) for i in keep_boxes_list]
        
        
        # img_show
        scale_show = 2 ** (level_output - level_show)
        # img = slide.read_region((0, 0), vis_level,
        #                 tuple([int(i / 2**vis_level) for i in slide.level_dimensions[0]])).convert('RGB')
        img_dyn = Image.open(os.path.join(args.prior_path, 'model_l1', 'save_l3', \
                                        os.path.basename(file).replace('.npy','_heat.png')))
        img_dyn_draw = ImageDraw.ImageDraw(img_dyn)
        boxes_dyn_show = [[int(i[0] * scale_show), int(i[1] * scale_show), \
                        int(i[2] * scale_show), int(i[3] * scale_show)] for i in boxes_save]
        for info in boxes_dyn_show:
            img_dyn_draw.rectangle(((info[0], info[1]), (info[2], info[3])), fill=None, outline='blue', width=1)
        img_dyn.save(os.path.join(args.output_path, os.path.basename(file).split('.')[0] + '_dyn.png'))
        
        img_fix = Image.open(os.path.join(args.prior_path, 'model_l1', 'save_l3', \
                                        os.path.basename(file).replace('.npy','_heat.png')))
        img_fix_draw = ImageDraw.ImageDraw(img_fix)
        boxes_fix_show = [[int((i[5] - args.ini_patchsize // 2) * scale_show), int((i[6] - args.ini_patchsize // 2) * scale_show), \
                       int((i[5] + args.ini_patchsize // 2) * scale_show), int((i[6] + args.ini_patchsize // 2) * scale_show)] for i in boxes_save]
        for info in boxes_fix_show:
            img_fix_draw.rectangle(((info[0], info[1]), (info[2], info[3])), fill=None, outline='blue', width=1)
        img_fix.save(os.path.join(args.output_path, os.path.basename(file).split('.')[0] + '_fix.png'))
        
        
        
