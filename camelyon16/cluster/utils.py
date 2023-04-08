import cv2
import os
import openslide
import json
import numpy as np
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm

def split_overlay_map1(grid, level):
    """
    Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
    :param grid: desnity mask to connect
    :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = [[0 for _ in range(n)] for _ in range(m)]
    count, queue, result = 0, [], []
    for i in range(m):
        for j in range(n):
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
                        if grid[i_cp][j_cp] == 255:
                            queue.append([i_cp, j_cp + 1])
                            queue.append([i_cp + 1, j_cp])
                            queue.append([i_cp, j_cp - 1])
                            queue.append([i_cp - 1, j_cp])
                    if pixel_area > (512 * 512) / (2**(6-level))**2:
                        break
                count += 1
                assert top < bot and left < right, "Coordination error!"
                pixel_area = (right - left + 1) * (bot - top + 1)
                result.append([count, (max(0, left), max(0, top)), (min(right, n), min(bot, m)), pixel_area])
                # compute pixel area by split_coord
    return result

def split_overlay_map(grid, level):
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
                if pixel_area > (1024 * 1024) / (2**(6-level))**2:
                    queue = []
                    break

            if  top < bot and left < right:
                count += 1
                result.append([count, (max(0, left), max(0, top)), (min(right, n), min(bot, m)), pixel_area])
            # compute pixel area by split_coord
    return result

def save_cropped_result1(img_array, window_size_threshold, level, density_prob_threshold, output_dir):
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
    for img_file in tqdm(img_array, total=len(img_array)):
        slide = openslide.OpenSlide(img_file)

        overlay_map = np.load(img_file.replace("tumor/", "dens_map_sliding_l3/").replace("tif", "npy"))
        overlay_map = Image.fromarray(overlay_map.transpose()).resize(slide.level_dimensions[6])
        overlay_map = ((np.asarray(overlay_map).transpose() / 255) > density_prob_threshold) * 255

        result = split_overlay_map(overlay_map, level)
        scale = [slide.level_dimensions[level][i] / slide.level_dimensions[6][i] for i in range(2)]
        result = [[i[0], (int(i[1][1] * scale[0]), int(i[1][0] * scale[1])), \
                            (int(i[2][1] * scale[0]), int(i[2][0] * scale[1]))] for i in result]
        
        # save dict
        new_result = []
        for info in result:
            pixel_area = (info[2][0] - info[1][0] + 1) * (info[2][1] - info[1][1] + 1)
            if pixel_area < window_size_threshold[0] * window_size_threshold[1]:
                continue
            else:
                new_result.append((info[1], info[2], pixel_area))
        save_dict.update({img_file: new_result})

        # # img_show
        # level_show = 4
        # img = slide.read_region((0, 0), level_show,
        #                 tuple([int(i / 2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        # img = img.resize(slide.level_dimensions[level_show])
        # img_draw = ImageDraw.ImageDraw(img)

        # scale_show = [slide.level_dimensions[level_show][i] / slide.level_dimensions[level][i] for i in range(2)]
        # result_show = [[i[0], (int(i[1][0] * scale_show[0]), int(i[1][1] * scale_show[1])), \
        #                     (int(i[2][0] * scale_show[0]), int(i[2][1] * scale_show[1]))] for i in result]
        # for info in result_show:
        #     pixel_area = (info[2][0] - info[1][0] + 1) * (info[2][1] - info[1][1] + 1)
        #     if pixel_area < (window_size_threshold[0] * window_size_threshold[1]) / (2**(level_show-level))**2:
        #         continue
        #     else:
        #         img_draw.rectangle((info[1],info[2]), fill=None, outline='blue', width=5)
        # img = img.resize(slide.level_dimensions[4])
        # img.save(os.path.join('/media/ps/passport2/hhy/camelyon16/train/crop_split_l{}/'.format(level), \
        #                          os.path.basename(img_file).split('.')[0] + '.png'))

    with open(os.path.join(output_dir, 'results.json'), 'w') as result_file:
        json.dump(save_dict, result_file)

def save_cropped_result(img_array, window_size_threshold, level, density_prob_threshold, output_dir):
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
    for img_file in tqdm(img_array, total=len(img_array)):
        slide = openslide.OpenSlide(img_file)

        overlay_map = np.load(img_file.replace("tumor/", "dens_map_sliding_l3/").replace("tif", "npy"))
        level_size = tuple([int(i / 2**6) for i in slide.level_dimensions[0]])
        overlay_map = Image.fromarray(overlay_map.transpose()).resize(level_size)
        overlay_map = ((np.asarray(overlay_map).transpose() / 255) > density_prob_threshold) * 255

        result = split_overlay_map(overlay_map, level)
        scale = 2 ** (6 - level)
        result_save = [[i[0], (int(i[1][1] * scale), int(i[1][0] * scale)), \
                            (int(i[2][1] * scale), int(i[2][0] * scale))] for i in result]
        
        # save dict
        new_result = []
        for info in result_save:
            pixel_area = (info[2][0] - info[1][0] + 1) * (info[2][1] - info[1][1] + 1)
            if pixel_area < window_size_threshold[0] * window_size_threshold[1]:
                continue
            else:
                new_result.append((info[1], info[2], pixel_area))
        save_dict.update({img_file: new_result})

        # img_show
        level_show = 4
        img = slide.read_region((0, 0), level_show,
                        tuple([int(i / 2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img_draw = ImageDraw.ImageDraw(img)

        scale_show = 2 ** (6 - level_show)
        result_show = [[i[0], (int(i[1][1] * scale_show), int(i[1][0] * scale_show)), \
                            (int(i[2][1] * scale_show), int(i[2][0] * scale_show))] for i in result]
        for info in result_show:
            pixel_area = (info[2][0] - info[1][0] + 1) * (info[2][1] - info[1][1] + 1)
            if pixel_area < (window_size_threshold[0] * window_size_threshold[1]) / (2**(level_show-level))**2:
                continue
            else:
                img_draw.rectangle((info[1],info[2]), fill=None, outline='blue', width=5)
        img = img.resize(slide.level_dimensions[4])
        img.save(os.path.join('/media/ps/passport2/hhy/camelyon16/train/crop_split_l{}/'.format(level), \
                                 os.path.basename(img_file).split('.')[0] + '.png'))

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
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = list(reversed(range(len(area))))  # np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    cluster_boxes = []
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
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

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

        cluster_dict = {'cluster_box':[int(xx1), int(yy1), int(xx2-xx1+1), int(yy2-yy1+1)]}
        child_list = []
        for box_idx in delete_idxs:
            child_box_x1 = x1[idxs[box_idx]]
            child_box_y1 = y1[idxs[box_idx]]
            child_box_w = x2[idxs[box_idx]] - x1[idxs[box_idx]] + 1
            child_box_h = y2[idxs[box_idx]] - y1[idxs[box_idx]] + 1
            child_list.append({'cluster_box': [int(child_box_x1), int(child_box_y1), int(child_box_w), int(child_box_h)]})
        cluster_dict.update({'child': child_list})
        cluster_boxes.append(cluster_dict)

        boxes[i,:4]=np.array([xx1,yy1,xx2,yy2])
        # delete all indexes from the index list that have
        idxs=np.delete(idxs,delete_idxs)

    # return only the bounding boxes that were picked using the
    # integer data type

    return boxes[pick].astype("int"), cluster_boxes
