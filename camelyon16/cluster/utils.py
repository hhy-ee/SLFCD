import cv2
import os
import openslide
import json
import time
import numpy as np
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm

def split_overlay_map1(grid, max_window_size, densmap_level, output_level):
    """
    Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
    :param grid: desnity mask to connect
    :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    x_idx, y_idx = np.array([]).astype(np.int64), np.array([]).astype(np.int64)
    for prob in np.unique(grid)[-1:0:-1]:
        idx = np.where(grid == prob)
        x_idx = np.concatenate((x_idx, idx[0]))
        y_idx = np.concatenate((y_idx, idx[1]))

    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = np.zeros((m, n))
    count, queue, result = 0, [], []
    for i, j in zip(x_idx, y_idx):
        if not visit[i][j]:
            if grid[i][j] == 0:
                visit[i][j] = 1
                continue
            queue.append([i, j])
            top, left = float("inf"), float("inf")
            bot, right = float("-inf"), float("-inf")
            while queue:
                i_cp, j_cp = queue.pop(0)
                top = min(i_cp, top)
                left = min(j_cp, left)
                bot = max(i_cp, bot)
                right = max(j_cp, right)
                if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                    visit[i_cp][j_cp] = 1
                    if grid[i_cp][j_cp] != 0:
                        queue.append([i_cp, j_cp + 1])
                        queue.append([i_cp + 1, j_cp])
                        queue.append([i_cp, j_cp - 1])
                        queue.append([i_cp - 1, j_cp])
                pixel_area = (right - left + 1) * (bot - top + 1)
                if pixel_area > (max_window_size[0] * max_window_size[1]) / (2**(densmap_level - output_level))**2:
                    queue = []
                    break

            if  top < bot and left < right:
                count += 1
                result.append([count, (max(0, left), max(0, top)), (min(right + 1, n), min(bot + 1, m)), pixel_area])
            # compute pixel area by split_coord
    return result

def point_based_split_overlay_map(grid, max_window_size, densmap_level, output_level):
    """
    Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
    :param grid: desnity mask to connect
    :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    x_idx, y_idx = np.array([]).astype(np.int64), np.array([]).astype(np.int64)
    for prob in np.unique(grid)[-1:0:-1]:
        idx = np.where(grid == prob)
        x_idx = np.concatenate((x_idx, idx[0]))
        y_idx = np.concatenate((y_idx, idx[1]))

    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = np.zeros((m, n))
    count, queue, result = 0, [], []
    for i, j in zip(x_idx, y_idx):
        cover = np.zeros((m, n))
        if not visit[i][j]:
            if grid[i][j] == 0:
                visit[i][j] = 1
                continue
            queue.append([i, j])
            top, left = float("inf"), float("inf")
            bot, right = float("-inf"), float("-inf")
            while queue:
                i_cp, j_cp = queue.pop(0)
                top = min(i_cp, top)
                left = min(j_cp, left)
                bot = max(i_cp, bot)
                right = max(j_cp, right)
                if 0 <= i_cp < m and 0 <= j_cp < n and not cover[i_cp][j_cp]:
                    cover[i_cp][j_cp] = 1
                    if grid[i_cp][j_cp] != 0:
                        queue.append([i_cp, j_cp + 1])
                        queue.append([i_cp + 1, j_cp])
                        queue.append([i_cp, j_cp - 1])
                        queue.append([i_cp - 1, j_cp])
                pixel_area = (right - left + 1) * (bot - top + 1)
                if max(right - left + 1, bot - top + 1) >= max_window_size[0] / 2**(densmap_level - output_level):
                    queue = []
                    break

            if  top < bot and left < right:
                count += 1
                result.append([count, (max(0, left), max(0, top)), (min(right + 1, n), min(bot + 1, m)), (j, i), pixel_area])
            # compute pixel area by split_coord
    return result

def patch_based_split_overlay_map(grid, max_window_size, densmap_level, output_level):
    """
    Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
    :param grid: desnity mask to connect
    :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    x_idx, y_idx = np.array([]).astype(np.int64), np.array([]).astype(np.int64)
    for prob in np.unique(grid)[-1:0:-1]:
        idx = np.where(grid == prob)
        x_idx = np.concatenate((x_idx, idx[0]))
        y_idx = np.concatenate((y_idx, idx[1]))

    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = np.zeros((m, n))
    count, queue, result = 0, [], []
    for i, j in zip(x_idx, y_idx):
        if not visit[i][j]:
            if grid[i][j] == 0:
                visit[i][j] = 1
                continue
            queue.append([i, j])
            top, left = float("inf"), float("inf")
            bot, right = float("-inf"), float("-inf")
            while queue:
                i_cp, j_cp = queue.pop(0)
                if right - left + 1 <= max_window_size[0] / 2**(densmap_level - output_level):
                    left = min(j_cp, left)
                    right = max(j_cp, right)
                    if 0 <= i_cp < m and 0 <= j_cp < n-1 and not visit[i_cp][j_cp]:
                        if grid[i_cp][j_cp] != 0:
                            queue.append([i_cp, j_cp - 1])
                            queue.append([i_cp, j_cp + 1])
                if bot - top + 1 <= max_window_size[1] / 2**(densmap_level - output_level):
                    top = min(i_cp, top)
                    bot = max(i_cp, bot)
                    if 0 <= i_cp < m-1 and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                        if grid[i_cp][j_cp] != 0:
                            queue.append([i_cp + 1, j_cp])
                            queue.append([i_cp - 1, j_cp])
                visit[i_cp][j_cp] = 1

            if  top < bot and left < right:
                count += 1
                pixel_area = (right - left + 1) * (bot - top + 1)
                result.append([count, (max(0, left), max(0, top)), (min(right + 1, n), min(bot + 1, m)), pixel_area])
            # compute pixel area by split_coord
    return result

def save_patch_based_cropped_result(img_array, max_window_size, min_window_size, dens_prob_thres, dens_dir, output_dir, 
                        densmap_level, output_level, vis_level=6):
    """
    A wrapper to conduct all necessary operation for generating density crops
    :param img_array: The input image to crop on
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dens_dir: The output dir to save density map
    :param output_img_dir: The output dir to save images
    :param output_anno_dir: The output dir to save annotations
    :param mode: The dataset to operate on (train/val/test)
    :return:
    """
    save_dict = {}
    time_total = 0.0
    for img_file in tqdm(img_array, total=len(img_array)):
        slide = openslide.OpenSlide(img_file)
        if not os.path.exists(os.path.join(dens_dir, os.path.basename(img_file).replace("tif", "npy"))):
            continue
        overlay_map = np.load(os.path.join(dens_dir, os.path.basename(img_file).replace("tif", "npy")))
        dens_shape = tuple([int(i / 2**densmap_level) for i in slide.level_dimensions[0]])
        overlay_map = cv2.resize(overlay_map, (dens_shape[1], dens_shape[0]), interpolation=cv2.INTER_CUBIC)
        overlay_map = ((overlay_map / 255) > dens_prob_thres) * overlay_map

        time_now = time.time()
        result = patch_based_split_overlay_map(overlay_map, max_window_size, densmap_level, output_level)
        time_total += time.time() - time_now

        scale_save = 2 ** (densmap_level - output_level)
        result_save = [[i[0], (int(i[1][1] * scale_save), int(i[1][0] * scale_save)), \
                            (int(i[2][1] * scale_save), int(i[2][0] * scale_save))] for i in result]
        
        # save dict
        new_result_save = []
        for info in result_save:
            if max(info[2][0]-info[1][0], info[2][1]-info[1][1]) > min_window_size:
                new_result_save.append(info)

        save_dict.update({img_file: new_result_save})

        # img_show
        scale_show = 2 ** (output_level - vis_level)
        # img = slide.read_region((0, 0), vis_level,
        #                 tuple([int(i / 2**vis_level) for i in slide.level_dimensions[0]])).convert('RGB')
        img = Image.open(os.path.join(dens_dir, os.path.basename(img_file).replace('.tif','_heat.png')))
        img_draw = ImageDraw.ImageDraw(img)
        result_show = [[i[0], (int(i[1][0] * scale_show), int(i[1][1] * scale_show)), \
                            (int(i[2][0] * scale_show), int(i[2][1] * scale_show))] for i in new_result_save]
        for info in result_show:
            img_draw.rectangle((info[1],info[2]), fill=None, outline='blue', width=1)
        img.save(os.path.join(output_dir, os.path.basename(img_file).split('.')[0] + '.png'))

    time_avg = time_total / len(img_array)
    print("Preprocessing -- crop split time: {} fps".format(time_avg))
    with open(os.path.join(output_dir, 'results.json'), 'w') as result_file:
        json.dump(save_dict, result_file)

def save_point_based_cropped_result(img_array, max_window_size, min_window_size, dens_prob_thres, dens_dir, output_dir, 
                        densmap_level, output_level, vis_level = 6):
    """
    A wrapper to conduct all necessary operation for generating density crops
    :param img_array: The input image to crop on
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dens_dir: The output dir to save density map
    :param output_img_dir: The output dir to save images
    :param output_anno_dir: The output dir to save annotations
    :param mode: The dataset to operate on (train/val/test)
    :return:
    """
    save_dict = {}
    time_total = 0.0
    for img_file in tqdm(img_array, total=len(img_array)):
        slide = openslide.OpenSlide(img_file)
        if os.path.exists(os.path.join(output_dir, os.path.basename(img_file).replace("tif", "png"))):
            continue
        overlay_map = np.load(os.path.join(dens_dir, os.path.basename(img_file).replace("tif", "npy")))
        dens_shape = tuple([int(i / 2**densmap_level) for i in slide.level_dimensions[0]])
        overlay_map = cv2.resize(overlay_map, (dens_shape[1], dens_shape[0]), interpolation=cv2.INTER_CUBIC)
        overlay_map = ((overlay_map / 255) > dens_prob_thres) * overlay_map

        time_now = time.time()
        result = point_based_split_overlay_map(overlay_map, max_window_size, densmap_level, output_level)
        time_total += time.time() - time_now

        scale_save = 2 ** (densmap_level - output_level)
        result_save = [[i[0], (int(i[1][1] * scale_save), int(i[1][0] * scale_save)), \
                              (int(i[2][1] * scale_save), int(i[2][0] * scale_save)), \
                              (int(i[3][1] * scale_save), int(i[3][0] * scale_save))] for i in result]
        
        # save dict
        new_result_save = []
        for info in result_save:
            if max(info[2][0]-info[1][0], info[2][1]-info[1][1]) > min_window_size:
                new_result_save.append((info[0], (info[3][0]-128, info[3][1]-128), (info[3][0]+128, info[3][1]+128)))

        save_dict.update({img_file: new_result_save})

        # img_show
        scale_show = 2 ** (output_level - vis_level)
        # img = slide.read_region((0, 0), vis_level,
        #                 tuple([int(i / 2**vis_level) for i in slide.level_dimensions[0]])).convert('RGB')
        img = Image.open(os.path.join(dens_dir, os.path.basename(img_file).replace('.tif','_heat.png')))
        img_draw = ImageDraw.ImageDraw(img)
        result_show = [[i[0], (int(i[1][0] * scale_show), int(i[1][1] * scale_show)), \
                            (int(i[2][0] * scale_show), int(i[2][1] * scale_show))] for i in new_result_save]
        for info in result_show:
            img_draw.rectangle((info[1],info[2]), fill=None, outline='blue', width=1)
        img.save(os.path.join(output_dir, os.path.basename(img_file).split('.')[0] + '.png'))

    time_avg = time_total / len(img_array)
    print("Preprocessing -- crop split time: {} fps".format(time_avg))
    with open(os.path.join(output_dir, 'results.json'), 'w') as result_file:
        json.dump(save_dict, result_file)

def generate_crop(img_file, level, slide, overlay_map, window_size_threshold=(70, 70)):
    """
    A wrapper to conduct all necessary operation for generating density crops
    :param img_array: The input image to crop on
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dens_dir: The output dir to save density map
    :param output_img_dir: The output dir to save images
    :param output_anno_dir: The output dir to save annotations
    :param mode: The dataset to operate on (train/val/test)
    :return:
    """

    result = split_overlay_map(overlay_map)
    scale = [slide.level_dimensions[3][i] / slide.level_dimensions[level][i] for i in range(2)]
    result = [[i[0], (int(i[1][1] * scale[0]), int(i[1][0] * scale[1])), \
                        (int(i[2][1] * scale[0]), int(i[2][0] * scale[1]))] for i in result]
    
    # img_show
    new_result = []
    for info in result:
        pixel_area = (info[2][0] - info[1][0] + 1) * (info[2][1] - info[1][1] + 1)
        if  pixel_area < window_size_threshold[0] * window_size_threshold[1]:
            continue
        else:
            new_result.append((info[1], info[2], pixel_area))

    return new_result


def ICM(boxes,iouthreshold,topN):
    """
    """
    boxes_num=len(boxes[:,0])
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    while boxes_num>topN:

        boxes=NMM(boxes, iouthreshold)
        boxes_num_after_merge=len(boxes[:,0])
        if boxes_num==boxes_num_after_merge:
            boxes_num_min=np.minimum(boxes_num_after_merge,topN)
            boxes=boxes[:boxes_num_min]
            break
        else:
            boxes_num=boxes_num_after_merge
    return boxes


def NMM(boxes, iouthreshold):
    """
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # sort the boxes according to score
    scores=boxes[:,4]
    idxs=np.argsort(-scores)
    boxes=boxes[idxs,:]
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = list(reversed(range(len(area))))  # np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    nmm_boxes = []
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # merge all boxes with overlap larger than threshold
        delete_idxs = np.concatenate(([last],
                                       np.where(overlap > iouthreshold)[0]))
        idxs=np.array(idxs)
        xx1 = np.amin(x1[idxs[delete_idxs]])
        yy1 = np.amin(y1[idxs[delete_idxs]])
        xx2 = np.amax(x2[idxs[delete_idxs]])
        yy2 = np.amax(y2[idxs[delete_idxs]])

        cluster_dict = {'cluster':[int(xx1), int(yy1), int(xx2-xx1), int(yy2-yy1)]}
        child_list = []
        for box_idx in delete_idxs:
            child_box_x1 = x1[idxs[box_idx]]
            child_box_y1 = y1[idxs[box_idx]]
            child_box_w = x2[idxs[box_idx]] - x1[idxs[box_idx]]
            child_box_h = y2[idxs[box_idx]] - y1[idxs[box_idx]]
            child_list.append({'cluster': [int(child_box_x1), int(child_box_y1), int(child_box_w), int(child_box_h)]})
        cluster_dict.update({'child': child_list})
        nmm_boxes.append(cluster_dict)

        boxes[i,:4]=np.array([xx1,yy1,xx2,yy2])
        # delete all indexes from the index list that have
        idxs=np.delete(idxs,delete_idxs)

    # return only the bounding boxes that were picked using the
    # integer data type

    return boxes[pick].astype("int"), nmm_boxes

def NMS(boxes, iouthreshold, score_refine=False):
    """
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # sort the boxes according to score
    scores=boxes[:,4]
    idxs=np.argsort(-scores)
    boxes=boxes[idxs,:]
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scr = boxes[:, 4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = list(reversed(range(len(area))))  # np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    nms_boxes = []
    count = 0
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[i]
        # overlap = (w * h) / (area[i] + area[idxs[:last]] - w * h)
        # merge all boxes with overlap larger than threshold
        delete_idxs = np.concatenate(([last],
                                       np.where(overlap > iouthreshold)[0]))
        supp_idxs = np.where(overlap > iouthreshold)[0]
        idxs=np.array(idxs)
        if score_refine:
            keep_dict = {'keep':[int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i]), 255.0+len(boxes)-count, scr[i]]}
        else:
            keep_dict = {'keep':[int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i]), scr[i]]}
        supp_list = []
        rege_list = []
        for box_idx in supp_idxs:
            supp_box_x1 = x1[idxs[box_idx]]
            supp_box_y1 = y1[idxs[box_idx]]
            supp_box_w = x2[idxs[box_idx]] - x1[idxs[box_idx]]
            supp_box_h = y2[idxs[box_idx]] - y1[idxs[box_idx]]
            supp_list.append([int(supp_box_x1), int(supp_box_y1), int(supp_box_w), int(supp_box_h), scr[idxs[box_idx]]])
            
            while (w[box_idx] * h[box_idx]) / area[i] > iouthreshold:
                x1[idxs[box_idx]] += 16
                x2[idxs[box_idx]] -= 16
                y1[idxs[box_idx]] += 16
                y2[idxs[box_idx]] -= 16
                w[box_idx] = np.maximum(0, np.minimum(x2[i], x2[idxs[box_idx]]) - np.maximum(x1[i], x1[idxs[box_idx]]))
                h[box_idx] = np.maximum(0, np.minimum(y2[i], y2[idxs[box_idx]]) - np.maximum(y1[i], y1[idxs[box_idx]]))
                
            rege_box_x1 = x1[idxs[box_idx]]
            rege_box_y1 = y1[idxs[box_idx]]
            rege_box_w = x2[idxs[box_idx]] - x1[idxs[box_idx]]
            rege_box_h = y2[idxs[box_idx]] - y1[idxs[box_idx]]
            rege_list.append([int(rege_box_x1), int(rege_box_y1), int(rege_box_w), int(rege_box_h), scr[idxs[box_idx]]])
            
        keep_dict.update({'supp': supp_list, 'rege': rege_list})
        nms_boxes.append(keep_dict)

        # boxes[i,:4]=np.array([xx1,yy1,xx2,yy2])
        # delete all indexes from the index list that have
        idxs=np.delete(idxs,delete_idxs)
        count += 1
    # return only the bounding boxes that were picked using the
    # integer data type

    return boxes[pick].astype("int"), nms_boxes