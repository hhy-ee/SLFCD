import os
import sys
import logging
import argparse
import openslide
import numpy as np
from tqdm import tqdm
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

    def __init__(self, tumor_path, tissue_path, slide, level, number):
        self.tumor_path = tumor_path
        self.tissue_path = tissue_path
        self.slide = slide
        self.level = level
        self.number = number

    def get_patch_point(self):
        # find tumor point
        mask_level = int(os.path.dirname(self.tumor_path).split('l')[-1])
        tissue_shape = tuple([int(i / 2**(mask_level-self.level)) for i in self.slide.level_dimensions[self.level]])
        if 'tumor' in os.path.basename(self.tumor_path):
            mask_tumor = Image.fromarray(np.load(self.tumor_path).transpose())
            mask_tumor = np.asarray(mask_tumor.resize(tissue_shape)).transpose()
        else:
            mask_tumor = np.zeros(tissue_shape) > 127
        X_idcs1, Y_idcs1 = np.where(mask_tumor)
        # find normal point
        mask_tissue = Image.fromarray(np.load(self.tissue_path).transpose()).resize(tissue_shape)
        mask_normal = np.asarray(mask_tissue).transpose() & (~ mask_tumor)
        X_idcs2, Y_idcs2 = np.where(mask_normal)

        centre_points_tumor = np.stack(np.vstack((X_idcs1.T, Y_idcs1.T)), axis=1)
        centre_points_normal = np.stack(np.vstack((X_idcs2.T, Y_idcs2.T)), axis=1)

        sampled_points_tumor = centre_points_tumor[np.random.choice(centre_points_tumor.shape[0],
                                    min(self.number, len(centre_points_tumor)), replace=False), :]
            
        sampled_points_normal = centre_points_normal[np.random.choice(centre_points_normal.shape[0],
                                                                    self.number, replace=False), :]

        sampled_points = np.concatenate((sampled_points_tumor, sampled_points_normal), axis=0)

        return (sampled_points * 2 ** (mask_level-self.level)).astype(np.int32)


def run(args):
    dir = os.listdir(args.tissue_path)
    for file in tqdm(sorted(dir), total=len(dir)):
        tumor_path = os.path.join(args.tumor_path, file.split('.')[0] + '.npy')
        tissue_path = os.path.join(args.tissue_path, file.split('.')[0] + '.npy')
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('_')[0], file.split('.')[0]+'.tif'))
        target_level = int(args.txt_path.split('l')[-1])
        sampled_points = patch_point_in_mask_gen(tumor_path, tissue_path, slide, target_level, args.patch_number).get_patch_point()

        mask_name = file.split(".")[0]
        name = np.full((sampled_points.shape[0], 1), mask_name)
        center_points = np.hstack((name, sampled_points))

        txt_path = os.path.join(args.txt_path, file.split('.')[0] + '.txt')

        with open(txt_path, "a") as f:
            np.savetxt(f, center_points, fmt="%s", delimiter=",")


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/train/",
        "/media/ps/passport2/hhy/camelyon16/train/tumor_mask_l5",
        "/media/ps/passport2/hhy/camelyon16/train/tissue_mask_l5",
        "/media/ps/passport2/hhy/camelyon16/train/sample_gen_l0",
        '1000'])
    run(args)


if __name__ == "__main__":
    main()
