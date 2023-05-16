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
    parser.add_argument('json_path', default=".", help='the path for box of ROI of json format')
    parser.add_argument('dens_path', default=".", help='the path for density map data of json format')
    parser.add_argument('label_path', help='The dir to annotations')
    parser.add_argument('output_path', help='The dir to save generated images and annotations')
    parser.add_argument('--iou_thres', type=float, help='Threshold defined to select the cropped region')
    parser.add_argument('--dens_thres', type=float, help='Threshold defined to generate embeddings')
    parser.add_argument('--save_label', type=bool, help='Threshold defined to generate embeddings')

    # args = parser.parse_args(['/media/ps/passport2/hhy/camelyon16/train/crop_split_l3/results.json', 
    #                           '/media/ps/passport2/hhy/camelyon16/train/dens_map_sliding_l3',
    #                           '/media/ps/passport2/hhy/camelyon16/train/tumor_mask_l3/',
    #                           '/media/ps/passport2/hhy/camelyon16/train/crop_cluster_l3'])
    
    # args = parser.parse_args(['/media/hy/hhy_data/camelyon16/train/crop_split_sliding_l1/results.json', 
    #                           '/media/hy/hhy_data/camelyon16/train/dens_map_sliding_l8',
    #                           '/media/hy/hhy_data/camelyon16/train/tumor_mask_l1/',
    #                           '/media/hy/hhy_data/camelyon16/train/crop_cluster_sliding_l1'])
    
    args = parser.parse_args(['/media/hy/hhy_data/camelyon16/test/crop_split_2048_base_l1/results.json', 
                              '/media/hy/hhy_data/camelyon16/test/dens_map_base_l8',
                              '/media/hy/hhy_data/camelyon16/test/tumor_mask_l1/',
                              '/media/hy/hhy_data/camelyon16/test/crop_cluster_2048_base_l1'])
    
    args.dens_thres = 0.5
    args.iou_thres = 0.3
    args.save_label = False
    return args


if __name__ == "__main__":

    args = parse_args()
    save_dict = {}
    time_total = 0.0

    # dens_level = int(args.dens_path.split('l')[-1])
    dens_level = 6
    output_level = int(args.output_path.split('l')[-1])

    with open(args.json_path, 'r') as f:
        candidates = json.load(f)
    
    for file_name in tqdm(candidates.keys(), total=len(candidates.keys())):
        
        # if os.path.exists(os.path.join(args.output_path, os.path.basename(file_name).replace('.tif','.png'))):
        #     continue

        # calculate score of each patches
        slide = openslide.OpenSlide(file_name)
        bbox_info = candidates[file_name]
        dens_map = np.load(os.path.join(args.dens_path, os.path.basename(file_name).replace('.tif','.npy')))
        size_scale = tuple([int(i / 2**dens_level) for i in slide.level_dimensions[0]])
        dens_map = cv2.resize(dens_map.transpose(), size_scale, interpolation=cv2.INTER_CUBIC).transpose() / 255
        scale = 2**(output_level - dens_level)
        boxes = []
        for box in bbox_info:
            left = int(box[1][0] * scale)
            right = int(box[2][0] * scale)
            top = int(box[1][1] * scale)
            bot = int(box[2][1] * scale)
            area = (right - left) * (bot - top)
            score = dens_map[left:right, top:bot].mean()
            boxes.append([box[1][0], box[1][1], box[2][0], box[2][1], score, box[3]])
        boxes = np.array(boxes)

        # preprocessing -- NMM
        time_now = time.time()
        if boxes.shape[0] == 0:
            continue
        cluster_boxes_list, cluster_boxes_dict = NMM(boxes, args.iou_thres)
        time_total += time.time() - time_now

        # add annotation
        if args.save_label:
            label_map = np.load(os.path.join(args.label_path, os.path.basename(file_name).replace('.tif','.npy')))
        for i, cluster in enumerate(cluster_boxes_dict):
            total_area, num_object = 0, 0
            clu_box = cluster['cluster_box']
            s_l = int(clu_box[0] * scale)
            s_r = int((clu_box[0]+clu_box[2]) * scale)
            s_t = int(clu_box[1] * scale)
            s_b = int((clu_box[1]+clu_box[3]) * scale)
            dens_patch = dens_map[s_l: s_r, s_t: s_b]
            dens_patch = cv2.resize(dens_patch, (clu_box[3], clu_box[2]), interpolation=cv2.INTER_CUBIC)

            for obj in dens_patch:
                total_area += int((obj > args.dens_thres).sum())
                if (obj > args.dens_thres).sum() > 0:
                    num_object += 1
            if num_object != 0:
                avg_area = total_area / num_object
            else:
                avg_area = total_area
            
            cluster.update({"total_area": total_area, "avg_area": avg_area, "num_object": num_object})
            if args.save_label:
                label_patch = label_map[clu_box[0]:clu_box[0]+clu_box[2], clu_box[1]:clu_box[1]+clu_box[3]]
                np.save(os.path.join(args.output_path, 'cluster_mask', \
                                    '{}_clu_{}.npy'.format(os.path.basename(file_name).split('.')[0], i)), label_patch)
            
            # # image illustration of cluster patches
            # o_s_x1 = int(clu_box[0] * slide.level_downsamples[output_level])
            # o_s_y1 = int(clu_box[1] * slide.level_downsamples[output_level])
            # img_patch_rgb = slide.read_region((o_s_x1, o_s_y1), output_level, (clu_box[2], clu_box[3])).convert('RGB')
            # label_patch_rgb = cv2.applyColorMap((label_patch*255).astype(np.uint8), cv2.COLORMAP_JET)
            # label_patch_rgb = cv2.cvtColor(label_patch_rgb, cv2.COLOR_BGR2RGB)
            # heat_img = cv2.addWeighted(label_patch_rgb.transpose(1,0,2), 0.5, np.asarray(img_patch_rgb), 0.5, 0)
            # cv2.imwrite(os.path.join(args.output_path, 'cluster_mask', \
            #                      '{}_clu_{}.png'.format(os.path.basename(file_name).split('.')[0], i)), heat_img)

            for j, child in enumerate(cluster['child']):
                total_area, num_object = 0, 0
                chi_box = child['cluster_box']
                s_l = int(chi_box[0] * scale)
                s_r = int((chi_box[0]+chi_box[2]) * scale)
                s_t = int(chi_box[1] * scale)
                s_b = int((chi_box[1]+chi_box[3]) * scale)
                dens_patch = dens_map[s_l: s_r, s_t: s_b]
                dens_patch = cv2.resize(dens_patch, (chi_box[3], chi_box[2]), interpolation=cv2.INTER_CUBIC)

                for obj in dens_patch:
                    total_area += int((obj > args.dens_thres).sum())
                    if (obj > args.dens_thres).sum() > 0:
                        num_object += 1
                if num_object != 0:
                    avg_area = total_area / num_object
                else:
                    avg_area = total_area

                child.update({"total_area": total_area, "avg_area": avg_area, "num_object": num_object})
                if args.save_label:
                    label_patch = label_map[chi_box[0]:chi_box[0]+chi_box[2], chi_box[1]:chi_box[1]+chi_box[3]]
                    np.save(os.path.join(args.output_path, 'cluster_mask', \
                                        '{}_clu_{}_chi_{}.npy'.format(os.path.basename(file_name).split('.')[0], i, j)), label_patch)
        
        # image illustration
        level_show = 6
        # img = slide.read_region((0, 0), level_show,
        #                 tuple([int(i / 2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img = Image.open(os.path.join(args.dens_path, os.path.basename(file_name).replace('.tif','_heat.png')))
        img_draw = ImageDraw.ImageDraw(img)
        scale_show = 2 ** (output_level - level_show)
        for cluster in cluster_boxes_dict:
            for child in cluster['child']:
                chi_box = child['cluster_box']
                img_draw.rectangle(((chi_box[0] * scale_show, chi_box[1] * scale_show), \
                    ((chi_box[0]-1+chi_box[2]) * scale_show, (chi_box[1]-1+chi_box[3]) * scale_show)), fill=None, outline='blue', width=1)
            clu_box = cluster['cluster_box']
            img_draw.rectangle(((clu_box[0] * scale_show, clu_box[1] * scale_show), \
                    ((clu_box[0]-1+clu_box[2]) * scale_show, (clu_box[1]-1+clu_box[3]) * scale_show)), fill=None, outline='green', width=1)
        img.save(os.path.join(args.output_path, os.path.basename(file_name).split('.')[0] + '.png'))

        save_dict.update({file_name.split('test/')[1]: cluster_boxes_dict})

    # save dict
    time_avg = time_total / len(candidates.keys())
    print("Preprocessing -- crop cluster time: {} fps".format(time_avg))
    with open(os.path.join(args.output_path, 'results.json'), 'w') as result_file:
        json.dump(save_dict, result_file)
