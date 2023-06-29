import cv2
import numpy as np
import scipy.stats.stats as st

from skimage.measure import label
from skimage.measure import regionprops
from openslide import OpenSlide
from openslide import OpenSlideUnsupportedFormatError

MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS = 0, 1, 2, 3, 4


class extractor_features(object):
    def __init__(self, probs_map, slide_patch):
        self._probs_map = probs_map / 255
        self._slide = get_image_open(slide_patch)

    def get_region_props(self, probs_map_threshold):
        labeled_img = label(probs_map_threshold)
        return regionprops(labeled_img, intensity_image=self._probs_map)

    def probs_map_set_p(self, threshold):
        probs_map_threshold = np.array(self._probs_map)

        probs_map_threshold[probs_map_threshold < threshold] = 0
        probs_map_threshold[probs_map_threshold >= threshold] = 1

        return probs_map_threshold

    def get_num_probs_region(self, region_probs):
        return len(region_probs)

    def get_tumor_region_to_tissue_ratio(self, region_props):
        tissue_area = cv2.countNonZero(self._slide)
        tumor_area = 0

        n_regions = len(region_props)
        for index in range(n_regions):
            tumor_area += region_props[index]['area']

        if tissue_area == 0:
            return 0.0
        return float(tumor_area) / tissue_area

    def get_largest_tumor_index(self, region_props):

        largest_tumor_index = -1

        largest_tumor_area = -1

        n_regions = len(region_props)
        for index in range(n_regions):
            if region_props[index]['area'] > largest_tumor_area:
                largest_tumor_area = region_props[index]['area']
                largest_tumor_index = index

        return largest_tumor_index

    def f_area_largest_tumor_region_t50(self):
        pass

    def get_average_prediction_across_tumor_regions(self, region_props):
        # close 255
        region_mean_intensity = [region.mean_intensity for region in region_props]
        return np.mean(region_mean_intensity)

    def get_feature(self, region_props, n_region, feature_name):
        feature = [0] * 5
        if n_region > 0:
            feature_values = [region[feature_name] for region in region_props]
            feature[MAX] = format_2f(np.max(feature_values))
            feature[MEAN] = format_2f(np.mean(feature_values))
            feature[VARIANCE] = format_2f(np.var(feature_values))
            feature[SKEWNESS] = format_2f(st.skew(np.array(feature_values)))
            feature[KURTOSIS] = format_2f(st.kurtosis(np.array(feature_values)))

        return feature


def compute_features(extractor, t=0.9):
    features = {}

    probs_map_threshold = extractor.probs_map_set_p(t)
    region_props = extractor.get_region_props(probs_map_threshold)
    
    # feature 1 ~ num object
    f_percentage_tumor_over_tissue_region = extractor.get_tumor_region_to_tissue_ratio(region_props)
    features.update({'ratio_tumor': f_percentage_tumor_over_tissue_region})

    # feature 2 ~ total area
    f_pixels_count_prob_gt = cv2.countNonZero(probs_map_threshold)  # 2
    features.update({'pixels_tumor': f_pixels_count_prob_gt})

    # feature 3 ~ avg area
    f_pixels_intensity_prob_gt = (extractor._probs_map *  probs_map_threshold).sum()
    features.update({'intensity_tumor': f_pixels_intensity_prob_gt})
    
    # # feature 4~
    # largest_tumor_region_index = extractor.get_largest_tumor_index(region_props)
    
    # f_intensity_largest_tumor_regions = region_props[largest_tumor_region_index].mean_intensity
    # features.update({'intensity_tumor_regions': f_intensity_largest_tumor_regions})

    # f_perimeter_tumor_regions = region_props[largest_tumor_region_index].perimeter
    # f_perimeter_probs_map = (probs_map_threshold.shape[0] + probs_map_threshold.shape[1]) * 2
    # f_perimeter_ratio_over_area = f_perimeter_tumor_regions / f_perimeter_probs_map
    # features.update({'perimeter_tumor_regions': f_perimeter_ratio_over_area})

    # f_extent_largest_tumor_regions = region_props[largest_tumor_region_index].extent
    # features.update({'extent_tumor_regions': f_extent_largest_tumor_regions})

    # f_solidity_largest_tumor_regions = region_props[largest_tumor_region_index].solidity
    # features.update({'solidity_tumor_regions': f_solidity_largest_tumor_regions})

    return features


def format_2f(number):
    return float("{0:.2f}".format(number))


def get_image_open(rgb_image):
    # hsv -> 3 channel
    rgb_image = np.array(rgb_image).transpose(1,0,2)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red)

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return image_open