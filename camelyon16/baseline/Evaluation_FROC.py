# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:09:32 2016
@author: Babak Ehteshami Bejnordi
Evaluation code for the Camelyon16 challenge on cancer metastases detecion
"""

import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
from tqdm import tqdm
import cv2
import os
import sys

def NMS(pred_mask, threshold, level_in, level_out, base_radius=3):
    prob_wsi, x_wsi, y_wsi = [], [], []
    X, Y = pred_mask.shape
    radius = base_radius * 2**(max(0, 8-level_in))
    while np.max(pred_mask / 255) > threshold:
        prob_max = pred_mask.max()
        max_idx = np.where(pred_mask == prob_max)
        x_mask, y_mask = max_idx[0][0], max_idx[1][0]
        prob_wsi.append(prob_max)
        x_wsi.append(int(x_mask * 2**(level_in-level_out)))
        y_wsi.append(int(y_mask * 2**(level_in-level_out)))
        x_min = x_mask - radius if x_mask - radius > 0 else 0
        x_max = x_mask + radius if x_mask + radius <= X else X
        y_min = y_mask - radius if y_mask - radius > 0 else 0
        y_max = y_mask + radius if y_mask + radius <= Y else Y
        
        pred_mask[x_min: x_max, y_min: y_max] = 0
    return prob_wsi, x_wsi, y_wsi
        

def computeEvaluationMask(maskDIR, resolution, level):
    """Computes the evaluation mask.

    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made

    Returns:
        evaluation_mask
    """

    pixelarray = (np.load(maskDIR) * 255).astype('uint')
    distance = nd.distance_transform_edt(255 - pixelarray)
    Threshold = 75 / (resolution * pow(2, level) * 2)  # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity=2)
    return evaluation_mask


def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)

    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object
        should be less than 275µm to be considered as ITC (Each pixel is
        0.243µm*0.243µm in level 0). Therefore the major axis of the object
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made

    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold = 275 / (resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i + 1)
    return Isolated_Tumor_Cells


def readCSVContent(csvDIR):
    """Reads the data inside CSV file

    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image

    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Xcorr, Ycorr, Probs = ([] for i in range(3))
    csv_lines = open(csvDIR, "r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        Probs.append(float(elems[0]))
        Xcorr.append(int(elems[1]))
        Ycorr.append(int(elems[2]))
    return Probs, Xcorr, Ycorr


def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells):
    """Generates true positive and false positive stats for the analyzed image

    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made

    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections

        TP_probs:   A list containing the probabilities of the True positive detections

        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)

        detection_summary:   A python dictionary object with keys that are the labels
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate].
        Lesions that are missed by the algorithm have an empty value.

        FP_summary:   A python dictionary object with keys that represent the
        false positive finding number and values that contain detection
        details [confidence score, X-coordinate, Y-coordinate].
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1, max_label + 1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []

    FP_counter = 0
    if (is_tumor):
        for i in range(0, len(Xcorr)):
            HittedLabel = evaluation_mask[Xcorr[i], Ycorr[i]]
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter += 1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i] > TP_probs[HittedLabel - 1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel - 1] = Probs[i]
    else:
        for i in range(0, len(Xcorr)):
            FP_probs.append(Probs[i])
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
            FP_counter += 1

    num_of_tumors = max_label - len(Isolated_Tumor_Cells)
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary


def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve

    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image

    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """

    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs) / float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs) / float(sum(FROC_data[3]))
    return total_FPs, total_sensitivity


def plotFROC(total_FPs, total_sensitivity):
    """Plots the FROC curve

    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds

    Returns:
        -
    """
    fig = plt.figure()
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000000')
    # plt.savefig('/media/ps/passport2/hhy/camelyon16/train/0.png')
    plt.show()


if __name__ == "__main__":
    
    # configuration
    wsi_folder = 'datasets/test/images'
    mask_folder = 'datasets/test/tumor_mask_l5'
    result_folder = 'datasets/test/prior_map_sampling_res50_o0.25_l1'
    # result_folder = 'datasets/test/dens_map_sampling_2s_l6/model_prior_o0.5_l1/save_cluster_roi_th_0.1_itc_th_1e0_1e9_nms_1.0_nmm_0.5_pack_dump_clu2_l1e-3_single_whole_region_fix_model_fix_size_l1'
    threshold = 0.5
    
    # default setting
    EVALUATION_MASK_LEVEL = int(mask_folder.split('l')[-1])  # Image level at which the evaluation is done
    # PREDICT_MASK_LEVEL = int(result_folder.split('l')[-1])
    PREDICT_MASK_LEVEL = 6
    L0_RESOLUTION = 0.243  # pixel resolution at level 0
    
    result_file_list = []
    result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.npy')]
    
    FROC_data = np.zeros((4, len(result_file_list)), dtype=object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=object)

    ground_truth_test = []
    ground_truth_test += [each[0:8] for each in os.listdir(mask_folder) if each.endswith('.npy')] # each[0:8] for test dataset
    ground_truth_test = set(ground_truth_test)

    caseNum = 0
    
    for case in tqdm(sorted(result_file_list), total=len(result_file_list)):
        # print('Evaluating Performance on image:', case[0:-4])
        # sys.stdout.flush()
        
        slide = openslide.open_slide(os.path.join(wsi_folder, case.split('.')[0] + '.tif'))
        result_mask = np.load(os.path.join(result_folder, case)) # 0~255 uint8
        
        if result_mask.dtype.name == 'float64':
            result_mask = (result_mask * 255).astype(np.uint8)
        scale = [int(i / 2**PREDICT_MASK_LEVEL) for i in slide.level_dimensions[0]]
        result_mask = cv2.resize(result_mask.astype(np.uint8), (scale[1], scale[0]), interpolation=cv2.INTER_CUBIC)
        
        Probs, Xcorr, Ycorr = NMS(result_mask, threshold, PREDICT_MASK_LEVEL, EVALUATION_MASK_LEVEL, base_radius=3)

        is_tumor = case[0:-4] in ground_truth_test
        if (is_tumor):
            maskDIR = os.path.join(mask_folder, case)
            evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
        else:
            evaluation_mask = 0
            ITC_labels = []

        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], \
        FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels)

        # # plot
        # s = 2 ** (EVALUATION_MASK_LEVEL - PREDICT_MASK_LEVEL)
        # result_folder_2s = './datasets/test/dens_map_sampling_2s_l6/model_distance_l1/save_roi_th_0.1_min_100_max_500_fix_size_256_non_holes_l3'
        # img_heat = cv2.imread(os.path.join(result_folder, case.split('.')[0]+'_heat.png'))
        # result_mask = np.load(os.path.join(result_folder_2s, case))
        # result_mask = cv2.resize(result_mask.astype(np.uint8), (scale[1], scale[0]), interpolation=cv2.INTER_CUBIC)
        # Probs, Xcorr, Ycorr = NMS(result_mask, threshold, PREDICT_MASK_LEVEL, EVALUATION_MASK_LEVEL, base_radius=3)
        # _, _, _, _, FP_summary_2s = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels)
        # for fp in FP_summary[1][caseNum]:
        #     patch_size = 8
        #     x, y = FP_summary[1][caseNum][fp][1], FP_summary[1][caseNum][fp][2]
        #     l, t, r, b = x*s-patch_size//2, y*s-patch_size//2, x*s+patch_size//2, y*s+patch_size//2
        #     cv2.rectangle(img_heat, (int(l), int(t)), (int(r), int(b)), (255,0,0), 1)
        # for fp in FP_summary_2s:
        #     patch_size = 12
        #     x, y = FP_summary_2s[fp][1], FP_summary_2s[fp][2]
        #     l, t, r, b = x*s-patch_size//2, y*s-patch_size//2, x*s+patch_size//2, y*s+patch_size//2
        #     cv2.rectangle(img_heat, (int(l), int(t)), (int(r), int(b)), (0,255,0), 1)
        # cv2.imwrite(os.path.join(os.path.dirname(result_folder), 'froc_result', case.split('.')[0] + '_fp.png'), img_heat)
        
        caseNum += 1

    # Compute FROC curve
    total_FPs, total_sensitivity = computeFROC(FROC_data)

    # plot FROC curve
    # plotFROC(total_FPs, total_sensitivity)

    eval_threshold = [.25, .5, 1, 2, 4, 8]
    eval_TPs = np.interp(eval_threshold, total_FPs[::-1], total_sensitivity[::-1])
    for i in range(len(eval_threshold)):
        print('Avg FP = ', str(eval_threshold[i]))
        print('Sensitivity = ', str(eval_TPs[i]))

    print('Avg Sensivity = ', np.mean(eval_TPs))
