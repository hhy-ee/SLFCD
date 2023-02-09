import os
import sys
import logging
import argparse

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")
parser.add_argument("tumor_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the tumor npy file")
parser.add_argument("normal_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the normal npy file")
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

    def __init__(self, tumor_path, normal_path, number):
        self.tumor_path = tumor_path
        self.normal_path = normal_path
        self.number = number

    def get_patch_point(self):
        mask_tumor = np.load(self.tumor_path)
        X_idcs1, Y_idcs1 = np.where(mask_tumor)
        mask_normal = np.load(self.normal_path)
        X_idcs2, Y_idcs2 = np.where(mask_normal)
        centre_points_tumor = np.stack(np.vstack((X_idcs1.T, Y_idcs1.T)), axis=1)
        centre_points_normal = np.stack(np.vstack((X_idcs2.T, Y_idcs2.T)), axis=1)
        sampled_points_label = np.zeros((self.number, 1))

        if centre_points_tumor.shape[0] > int(self.number / 2):
            sampled_points_tumor = centre_points_tumor[np.random.randint(centre_points_tumor.shape[0],
                                                                    size=int(self.number / 2)), :]
            sampled_points_normal = centre_points_normal[np.random.randint(centre_points_normal.shape[0],
                                                                    size=self.number - int(self.number / 2)), :]
            sampled_points_label[:int(self.number / 2)] = 1
        else:
            sampled_points_tumor = centre_points_tumor[np.random.randint(centre_points_tumor.shape[0],
                                                                    size=len(centre_points_tumor)), :]
            sampled_points_normal = centre_points_normal[np.random.randint(centre_points_normal.shape[0],
                                                                    size=self.number - len(centre_points_tumor)), :]
            sampled_points_label[:len(centre_points_tumor)] = 1

        return np.concatenate((sampled_points_tumor, sampled_points_normal), axis=0), sampled_points_label


def run(args):
    dir = os.listdir(args.normal_path)
    for file in dir:
        tumor_path = os.path.join(args.tumor_path, file.split('.')[0] + '.npy')
        normal_path = os.path.join(args.normal_path, file.split('.')[0] + '.npy')
        sampled_points, label = patch_point_in_mask_gen(tumor_path, normal_path, args.patch_number).get_patch_point()
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
        "/media/ps/passport2/hhy/camelyon16/training/tumor_mask_l6",
        "/media/ps/passport2/hhy/camelyon16/training/normal_mask_l6",
        "/media/ps/passport2/hhy/camelyon16/training/sample_gen",
        '1000'])
    run(args)


if __name__ == "__main__":
    main()
