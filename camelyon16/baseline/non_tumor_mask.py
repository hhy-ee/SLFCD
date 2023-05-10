import sys
import os
import argparse
import logging

import numpy as np

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

parser = argparse.ArgumentParser(description="Get the normal region"
                                             " from tumor WSI ")
parser.add_argument("tumor_path", default=None, metavar='TUMOR_PATH', type=str,
                    help="Path to the tumor mask npy")
parser.add_argument("tissue_path", default=None, metavar='TISSUE_PATH', type=str,
                    help="Path to the tissue mask npy")
parser.add_argument("normal_path", default=None, metavar='NORMAL_PATCH', type=str,
                    help="Path to the output normal region from tumor WSI npy")


def run(args):
    dir = os.listdir(args.tumor_path)
    for file in tqdm(sorted(dir), total=len(dir)):
        if file.split('.')[-1] == 'npy':
            tumor_mask = np.load(os.path.join(args.tumor_path, file))
            tissue_mask = np.load(os.path.join(args.tissue_path, file))

            normal_mask = tissue_mask & (~ tumor_mask)

            np.save(os.path.join(args.normal_path, file), normal_mask)

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/training/tumor_mask_l6",
        "/media/ps/passport2/hhy/camelyon16/training/tissue_mask_l6",
        "/media/ps/passport2/hhy/camelyon16/training/normal_mask_l6"])
    run(args)


if __name__ == "__main__":
    main()
