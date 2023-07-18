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
from camelyon16.cluster.utils import NMS, NMM
from camelyon16.cluster.probs_ops import extractor_features, compute_features

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
    parser.add_argument('--roi_threshold', help='type of the generator of the first stage')
    parser.add_argument('--itc_threshold', help='The threshold of ITC to be eliminated')
    parser.add_argument('--ini_patchsize', help='The size of initial patch size')
    parser.add_argument('--nmm_threshold', help='The threshold to select the cropped region')
    parser.add_argument('--fea_threshold', help='The threshold to generate feature')
    parser.add_argument('--image_show', default=True, help='whether to visualization')
    parser.add_argument('--label_save', default=True, help='whether to visualization')

    args = parser.parse_args(['./datasets/train/tumor', 
                              './datasets/train/dens_map_sampling_l8',
                              './datasets/train/crop_split_new_l1'])
    args.roi_threshold = 0.1
    args.itc_threshold = (16, 2500)
    args.ini_patchsize = 256
    args.nms_threshold = 0.5
    args.nmm_threshold = 0.1
    args.fea_threshold = 0.5
    args.image_show = True
    args.label_save = False
    return args

if __name__ == "__main__":

    args = parse_args()

    time_total = 0.0
    level_show = 6
    level_prior = 6
    level_input = 3
    level_output = int(args.output_path.split('l')[-1])

    scale_show = 2 ** (level_output - level_show)
    scale_in = 2 ** (level_prior - level_input)
    scale_out = 2 ** (level_input - level_output)
    scale_feature = 2 ** (level_prior - level_output)
    
    dyn_boxes_dict = {}
    final_boxes_dict ={}
    
    dir = os.listdir(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l6'))
    for file in tqdm(sorted(dir), total=len(dir)):
        # initialization
        filtered_properties = []
        total_boxes_dyn = []
        total_boxes_nms = []
        total_boxes_nmm = []
        
        # calculate score of each patches
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        first_stage_map = np.load(os.path.join(args.prior_path, 'model_l1/save_l3', file))
        prior_shape = tuple([int(i / 2**level_prior) for i in slide.level_dimensions[0]])
        prior_map = cv2.resize(first_stage_map, (prior_shape[1], prior_shape[0]), interpolation=cv2.INTER_CUBIC)
        if 'train' in args.wsi_path:
            gt_tumor_mask = np.load(os.path.join(os.path.dirname(args.wsi_path), 'tumor_mask_l{}'.format(level_prior), file))
            prior_map = prior_map * gt_tumor_mask
        POI = (prior_map / 255) > args.roi_threshold
        
        # Computes the inference mask
        filled_image = nd.morphology.binary_fill_holes(POI)
        evaluation_mask = measure.label(filled_image, connectivity=2)
        
        # eliminate abnormal tumor cells
        max_label = np.amax(evaluation_mask)
        properties = measure.regionprops(evaluation_mask)
        filled_mask = np.zeros(filled_image.shape) > 0
        conf_map = np.zeros(filled_image.shape).astype(np.uint8)
        for i in range(0, max_label):
            patch_num = properties[i].image_filled.sum()
            if patch_num > args.itc_threshold[0] and patch_num < args.itc_threshold[1]:
                l, t, r, b = properties[i].bbox
                filled_mask[l: r, t: b] = np.logical_or(filled_mask[l: r, t: b], properties[i].image_filled)
                filtered_properties.append(properties[i])
                region_confidence = prior_map[properties[i].coords[:,0], properties[i].coords[:,1]].mean()
                conf_map[properties[i].coords[:,0], properties[i].coords[:,1]] = region_confidence
        
        if args.image_show:
            img_rgb = slide.read_region((0, 0), level_show, \
                        tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
            img_rgb = np.asarray(img_rgb).transpose((1,0,2))
            show_shape = tuple([int(i / 2**level_show) for i in slide.level_dimensions[0]])
            prior_show = cv2.resize(prior_map, (show_shape[1], show_shape[0]), interpolation=cv2.INTER_CUBIC)
            prior_rgb = cv2.applyColorMap(prior_show.astype(np.uint8), cv2.COLORMAP_JET)
            prior_rgb = cv2.cvtColor(prior_rgb, cv2.COLOR_BGR2RGB)
            heat_img = cv2.addWeighted(prior_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
            img = Image.fromarray(heat_img)
            img_draw = ImageDraw.ImageDraw(img)
            for i in range(len(filtered_properties)):
                l, t, r, b  = filtered_properties[i].bbox
                img_draw.rectangle(((l, t), (r, b)), fill=None, outline='blue', width=1)
            heat_img_rect = np.asarray(img)
            cv2.imwrite(os.path.join(args.output_path, file.split('.')[0] + '_ctc.png'), heat_img_rect)
            
            distance, coord = nd.distance_transform_edt(filled_mask, return_indices=True)
            edge_X, edge_Y = np.where(distance == 1)
            prior_heat = cv2.imread(os.path.join(args.prior_path, 'model_l1/save_l3', file.replace('.npy','_heat.png')))
            prior_heat[edge_Y, edge_X, :] = [0, 255, 0]
            cv2.imwrite(os.path.join(args.output_path, file.split('.')[0] + '_edge.png'), prior_heat)
            
            # plot 
            img_rgb = slide.read_region((0, 0), level_show, \
                                tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
            img_rgb = np.asarray(img_rgb).transpose((1,0,2))
            conf_map_res = cv2.resize(conf_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
            conf_map_rgb = cv2.applyColorMap(conf_map_res, cv2.COLORMAP_JET)
            conf_map_rgb = cv2.cvtColor(conf_map_rgb, cv2.COLOR_BGR2RGB)
            heat_img = cv2.addWeighted(conf_map_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
            cv2.imwrite(os.path.join(args.output_path, file.split('.')[0] + '_conf.png'), heat_img)
            
        # generate patches from each tumor cell
        distance, coord = nd.distance_transform_edt(filled_mask, return_indices=True)
        ext_shape = tuple([int(i / 2**level_output) for i in slide.level_dimensions[0]])
        feature_map = cv2.resize(first_stage_map, (ext_shape[1], ext_shape[0]), interpolation=cv2.INTER_CUBIC) 
        for i in range(len(filtered_properties)):
            boxes_tumor = []
            tc_X = filtered_properties[i].coords[:,0]
            tc_Y = filtered_properties[i].coords[:,1]
            for idx in range(len(tc_X)):
                x_center, y_center = tc_X[idx], tc_Y[idx]
                patch_size = args.ini_patchsize // scale_out
                l = x_center * scale_in - patch_size // 2
                t = y_center * scale_in - patch_size // 2
                r, b = l + patch_size, t + patch_size
                pos_idx = np.where(first_stage_map[l: r, t: b] / 255 > args.roi_threshold)
                scr = first_stage_map[l: r, t: b][pos_idx].mean()
                l, t, r, b = l * scale_out, t * scale_out, r * scale_out, b  * scale_out
                boxes_tumor.append([l, t, r, b, scr])
            
            # save fix-sized patches
            # boxes_save = [{'keep': [int(i[0]), int(i[1]), int(i[2] - i[0]), int(i[3] - i[1])]} for i in boxes_tumor]
            # dyn_boxes_dict.update({'{}_tc_{}'.format(file.split('.npy')[0], i): boxes_save})

            # dynamic patches 1
            boxes_dyn = []
            for idx in range(len(tc_X)):
                x_center, y_center = tc_X[idx], tc_Y[idx]
                edge_dist = distance[x_center, y_center]
                dyn_patch_size = max(args.ini_patchsize // edge_dist, 64)
                dyn_patch_size = int(dyn_patch_size // scale_out)
                l = x_center * scale_in - dyn_patch_size // 2
                t = y_center * scale_in - dyn_patch_size // 2
                r, b = l + dyn_patch_size, t + dyn_patch_size
                l, t, r, b = l * scale_out, t * scale_out, r * scale_out, b  * scale_out
                boxes_dyn.append([l, t, r, b, boxes_tumor[idx][-1]])


            # dynamic patches 2
            # boxes_tumor = np.array(boxes_tumor)
            # _, nms_boxes_dict = NMS(boxes_tumor, args.nms_threshold, box_shrink=True)
            # boxes_dyn = [i['keep'] for i in nms_boxes_dict] + [i for j in nms_boxes_dict for i in j['rege']]
            # first_nms_boxes_dict = nms_boxes_dict
            # while len(nms_boxes_dict) != len(boxes_tumor):
            #     boxes_dyn = [i['keep'] for i in nms_boxes_dict] + [i for j in nms_boxes_dict for i in j['rege']]
            #     boxes_re_nms = np.array([[i[0], i[1], i[0] + i[2], i[1] + i[3], i[4]] for i in boxes_dyn])
            #     _, nms_boxes_dict = NMS(boxes_re_nms, args.nms_threshold, box_shrink=True)
            # boxes_dyn = [[i[0], i[1], i[0]+i[2], i[1]+i[3], i[4]] for i in boxes_dyn]
            # boxes_dyn = [i['keep'][:4] + [i['keep'][5]] for i in first_nms_boxes] + boxes_dyn[len(first_nms_boxes):]

            total_boxes_dyn += boxes_dyn

            # save dynamic-sized patches
            boxes_save = [{'keep': [int(i[0]), int(i[1]), int(i[2] - i[0]), int(i[3] - i[1])]} for i in boxes_dyn]
            dyn_boxes_dict.update({'{}_tc_{}'.format(file.split('.npy')[0], i): boxes_save})

            # NMS
            # boxes_dyn = np.array(boxes_dyn)
            # keep_boxes_list, nms_boxes_dict = NMS(boxes_dyn, args.nms_threshold)
            # boxes_nms = [list(i) for i in keep_boxes_list]
            # total_boxes_nms += boxes_nms
            
            boxes_nms = np.array(boxes_dyn)

            # NMM
            boxes_nms = np.array(boxes_nms)
            cluster_boxes_list, nmm_boxes_dict = NMM(boxes_nms, args.nmm_threshold)
            total_boxes_nmm += nmm_boxes_dict
            
            # feature extraction
            tc_bbox = [c * scale_feature for c in filtered_properties[i].bbox]
            tc_w, tc_h = tc_bbox[2] - tc_bbox[0], tc_bbox[3] - tc_bbox[1]
            tc_l = int(tc_bbox[0] * slide.level_downsamples[level_output])
            tc_t = int(tc_bbox[1] * slide.level_downsamples[level_output])
            tc_slide = slide.read_region((tc_l, tc_t), level_output, (tc_w, tc_h))
            tc_map = feature_map[tc_bbox[0]:tc_bbox[2], tc_bbox[1]:tc_bbox[3]]
            extractor = extractor_features(tc_map, tc_slide)
            tc_features = compute_features(extractor, args.fea_threshold)
            tc_features.update({'height': tc_h, 'width': tc_w})
            nmm_boxes_dict = [tc_features] + nmm_boxes_dict
            for j, cluster in enumerate(nmm_boxes_dict[1:]):
                clu_box = cluster['cluster']
                dens_patch = feature_map[clu_box[0]: clu_box[0]+clu_box[2], clu_box[1]: clu_box[1]+clu_box[3]]
                slide_patch = slide.read_region((int(clu_box[0]* slide.level_downsamples[level_output]), \
                                                int(clu_box[1]* slide.level_downsamples[level_output])), \
                                                level_output, (clu_box[2], clu_box[3]))
                # # feature
                extractor = extractor_features(dens_patch, slide_patch)
                features = compute_features(extractor, args.fea_threshold)
                cluster.update(features)
                
                # plot & save
                if args.image_show:
                    img_patch_rgb = np.asarray(slide_patch.convert('RGB')).transpose((1,0,2))
                    label_patch_rgb = cv2.applyColorMap(dens_patch, cv2.COLORMAP_JET)
                    label_patch_rgb = cv2.cvtColor(label_patch_rgb, cv2.COLOR_BGR2RGB)
                    heat_img = cv2.addWeighted(label_patch_rgb, 0.5, img_patch_rgb, 0.5, 0)
                    binary_map = (extractor.probs_map_set_p(args.fea_threshold) *255).astype(np.uint8)
                    cv2.imwrite(os.path.join(args.output_path, os.path.basename(file).split('.')[0] + '_patch.png'), heat_img)
                    cv2.imwrite(os.path.join(args.output_path, os.path.basename(file).split('.')[0] + '_binary.png'), binary_map)
                if args.label_save:
                    np.save(os.path.join(args.output_path, 'cluster_mask', '{}_tc_{}_clu_{}.npy'.\
                                        format(os.path.basename(file).split('.')[0], i, j)), dens_patch)
                    
                for k, child in enumerate(cluster['child']):
                    chi_box = child['cluster']
                    dens_patch = feature_map[chi_box[0]: chi_box[0]+chi_box[2], chi_box[1]: chi_box[1]+chi_box[3]]
                    slide_patch = slide.read_region((int(chi_box[0]* slide.level_downsamples[level_output]), \
                                                int(chi_box[1]* slide.level_downsamples[level_output])), \
                                                level_output, (chi_box[2], chi_box[3])) 
                    # # feature
                    extractor = extractor_features(dens_patch, slide_patch)
                    features = compute_features(extractor, args.fea_threshold)
                    child.update(features)
                
                    # plot & save
                    if args.image_show:
                        img_patch_rgb = np.asarray(slide_patch.convert('RGB')).transpose((1,0,2))
                        label_patch_rgb = cv2.applyColorMap(dens_patch, cv2.COLORMAP_JET)
                        label_patch_rgb = cv2.cvtColor(label_patch_rgb, cv2.COLOR_BGR2RGB)
                        heat_img = cv2.addWeighted(label_patch_rgb, 0.5, img_patch_rgb, 0.5, 0)
                        binary_map = (extractor.probs_map_set_p(args.fea_threshold) *255).astype(np.uint8)
                        cv2.imwrite(os.path.join(args.output_path, os.path.basename(file).split('.')[0] + '_patch.png'), heat_img)
                        cv2.imwrite(os.path.join(args.output_path, os.path.basename(file).split('.')[0] + '_binary.png'), binary_map)
                    if args.label_save: 
                        np.save(os.path.join(args.output_path, 'cluster_mask', '{}_tc_{}_clu_{}_chi_{}.npy'.\
                                            format(os.path.basename(file).split('.')[0], i, j, k)), dens_patch)

            
            final_boxes_dict.update({'{}_tc_{}'.format(file.split('.npy')[0], i): nmm_boxes_dict})
        
        if args.image_show:
            img = Image.open(os.path.join(args.prior_path, 'model_l1/save_l3', file.replace('.npy','_heat.png')))
            img_nms_draw = ImageDraw.ImageDraw(img)
            boxes_nms_show = [[int(i[0] * scale_show), int(i[1] * scale_show), \
                            int(i[2] * scale_show), int(i[3] * scale_show)] for i in total_boxes_nms]
            for info in boxes_nms_show:
                img_nms_draw.rectangle(((info[0], info[1]), (info[2], info[3])), fill=None, outline='blue', width=1)
            img.save(os.path.join(args.output_path, os.path.basename(file).split('.')[0] + '_nms.png'))
            
            img = Image.open(os.path.join(args.prior_path, 'model_l1/save_l3', file.replace('.npy','_heat.png')))
            img_dyn_draw = ImageDraw.ImageDraw(img)
            boxes_dyn_show = [[int(i[0] * scale_show), int(i[1] * scale_show), \
                            int(i[2] * scale_show), int(i[3] * scale_show)] for i in total_boxes_dyn]
            for info in boxes_dyn_show:
                img_dyn_draw.rectangle(((info[0], info[1]), (info[2], info[3])), fill=None, outline='blue', width=1)
            img.save(os.path.join(args.output_path, os.path.basename(file).split('.')[0] + '_dyn.png'))

            img = Image.open(os.path.join(args.prior_path, 'model_l1/save_l3', file.replace('.npy','_heat.png')))
            img_draw = ImageDraw.ImageDraw(img)
            for cluster in total_boxes_nmm:
                for child in cluster['child']:
                    chi_box = child['cluster']
                    img_draw.rectangle(((chi_box[0] * scale_show, chi_box[1] * scale_show), 
                                        ((chi_box[0]-1+chi_box[2]) * scale_show, \
                                        (chi_box[1]-1+chi_box[3]) * scale_show)), \
                                        fill=None, outline='blue', width=1)
                clu_box = cluster['cluster']
                img_draw.rectangle(((clu_box[0] * scale_show, clu_box[1] * scale_show), \
                                    ((clu_box[0]-1+clu_box[2]) * scale_show, \
                                    (clu_box[1]-1+clu_box[3]) * scale_show)), \
                                    fill=None, outline='green', width=1)
            img.save(os.path.join(args.output_path, os.path.basename(file).split('.')[0] + '_nmm.png'))
        
    with open(os.path.join(args.output_path, 'results.json'), 'w') as result_file:
        json.dump(final_boxes_dict, result_file)
    with open(os.path.join(args.output_path, 'results_boxes.json'), 'w') as result_file:
        json.dump(dyn_boxes_dict, result_file)