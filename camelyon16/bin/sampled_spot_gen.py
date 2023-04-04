import os
import sys
import logging
import argparse
import openslide
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")
parser.add_argument("wsi_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the wsi tif file")
parser.add_argument("tumor_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the tumor npy file")
parser.add_argument("tissue_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the tissue npy file")
parser.add_argument("txt_path", default=None, metavar="TXT_PATH", type=str,
                    help="Path to the txt file")
parser.add_argument("patch_number", default=None, metavar="PATCH_NUMB", type=int,
                    help="The number of patches extracted from WSI")
parser.add_argument("--level", default=6, metavar="LEVEL", type=int,
                    help="Bool format, whether or not")


class patch_point_in_mask_gen(object):
    '''
    extract centre point from mask
    inputs: mask path, centre point number
    outputs: centre point
    '''

    def __init__(self, tumor_path, tissue_path, info, number):
        self.tumor_path = tumor_path
        self.tissue_path = tissue_path
        self.name = info[0]
        self.size = info[1]
        self.number = number
        
    def get_patch_point(self):
        # find tumor point
        if 'tumor' in self.name:
            mask_tumor = np.load(self.tumor_path)
        else:
            mask_tumor = np.zeros(self.size) > 127
        X_idcs1, Y_idcs1 = np.where(mask_tumor)
        # find normal point
        mask_tissue = Image.fromarray(np.load(self.tissue_path).transpose()).resize(self.size)
        mask_normal = np.asarray(mask_tissue).transpose() & (~ mask_tumor)
        X_idcs2, Y_idcs2 = np.where(mask_normal)
        # find invalid point   
        mask_invalid = ~ np.asarray(mask_tissue).transpose()
        X_idcs3, Y_idcs3 = np.where(mask_invalid)

        centre_points_tumor = np.stack(np.vstack((X_idcs1.T, Y_idcs1.T)), axis=1)
        centre_points_normal = np.stack(np.vstack((X_idcs2.T, Y_idcs2.T)), axis=1)
        centre_points_invalid = np.stack(np.vstack((X_idcs3.T, Y_idcs3.T)), axis=1)
        sampled_points_label = np.zeros((self.number, 1))

        if centre_points_tumor.shape[0] > int(self.number / 2):
            sampled_points_tumor = centre_points_tumor[np.random.choice(centre_points_tumor.shape[0],
                                                                    int(self.number / 2), replace=False), :]
            sampled_points_normal = centre_points_normal[np.random.choice(centre_points_normal.shape[0],
                                                                    self.number-int(self.number / 2)-self.number//10, replace=False), :]
            sampled_points_invalid = centre_points_invalid[np.random.choice(centre_points_invalid.shape[0],
                                                                    self.number//10, replace=False), :]
            sampled_points_label[:int(self.number / 2)] = 1
        else:
            sampled_points_tumor = centre_points_tumor[np.random.choice(centre_points_tumor.shape[0],
                                                                    len(centre_points_tumor), replace=False), :]
            sampled_points_normal = centre_points_normal[np.random.choice(centre_points_normal.shape[0],
                                                                    self.number-len(centre_points_tumor)-self.number//10, replace=False), :]
            sampled_points_invalid = centre_points_invalid[np.random.choice(centre_points_invalid.shape[0],
                                                                    self.number//10, replace=False), :]
            sampled_points_label[:len(centre_points_tumor)] = 1

        return np.concatenate((sampled_points_tumor, sampled_points_normal, sampled_points_invalid), axis=0), sampled_points_label


def run(args):
    dir = os.listdir(args.tissue_path)
    for file in dir:
        tumor_path = os.path.join(args.tumor_path, file.split('.')[0] + '.npy')
        tissue_path = os.path.join(args.tissue_path, file.split('.')[0] + '.npy')
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('_')[0], file.split('.')[0]+'.tif'))
        info  = (file.split('.')[0], slide.level_dimensions[args.level])
        sampled_points, label = patch_point_in_mask_gen(tumor_path, tissue_path, info, args.patch_number).get_patch_point()
        sampled_points, label = sampled_points.astype(np.int32), label.astype(np.int32) # make sure the factor

        mask_name = file.split(".")[0]
        name = np.full((sampled_points.shape[0], 1), mask_name)
        center_points = np.hstack((name, sampled_points, label))

        txt_path = os.path.join(args.txt_path, file.split('.')[0] + '.txt')

        with open(txt_path, "a") as f:
            np.savetxt(f, center_points, fmt="%s", delimiter=",")


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/training/",
        "/media/ps/passport2/hhy/camelyon16/training/tumor_mask_l6",
        "/media/ps/passport2/hhy/camelyon16/training/tissue_mask_l6",
        "/media/ps/passport2/hhy/camelyon16/training/sample_gen_l6",
        '1000'])
    args.level = 6
    run(args)


if __name__ == "__main__":
    main()
