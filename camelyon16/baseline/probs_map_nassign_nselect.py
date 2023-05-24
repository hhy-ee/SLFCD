import sys
import os
import argparse
import logging
import json
import time
import cv2
from PIL import Image
from skimage import transform

import numpy as np
import torch
import openslide
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from camelyon16.cluster.utils import generate_crop

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from camelyon16.data.prob_producer_base_nassign_nselect import WSIPatchDataset  # noqa


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('assign_path', default=None, metavar='ASSIGN_PATH', type=str,
                    help='Path to the json file related to assignment of patch')
parser.add_argument('--GPU', default='2', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=0, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--subdivisions', default=0, type=int, help='whether to'
                    'use slide window paradigm with overlap.')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')

def chose_model(mod):
    if mod == 'segmentation':
        model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
    else:
        raise Exception("I have not add any models. ")
    return model


def get_probs_map(model, slide, level, dataloader):

    probs_map = np.zeros(tuple([int(i / 2**level) for i in slide.level_dimensions[0]]))
    counter = np.zeros(tuple([int(i / 2**level) for i in slide.level_dimensions[0]]))
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    time_total = 0.
    
    with torch.no_grad():
        for (data, box, resize) in dataloader:
            data = Variable(data.cuda(non_blocking=True))
            output = model(data)
            # because of torch.squeeze at the end of forward in resnet.py, if the
            # len of dim_0 (batch_size) of data is 1, then output removes this dim.
            # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
            probs = output['out'][:, :].sigmoid().cpu().data.numpy()
            for i in range(probs.shape[0]):
                counter[box[0][i]:box[2][i], box[1][i]:box[3][i]] += 1
                patch_prob = probs[i, 0, :]
                if resize[i]:
                    patch_prob = transform.resize(probs[i, 0, :], (box[2][i]-box[0][i], box[3][i]-box[1][i]))
                probs_map[box[0][i]:box[2][i], box[1][i]:box[3][i]] += patch_prob
            count += 1
            time_spent = time.time() - time_now
            time_now = time.time()
            logging.info(
                '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
                .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                    dataloader.dataset._rotate, count, num_batch, time_spent))
            time_total += time_spent
        
        zero_mask = counter == 0
        probs_map[~zero_mask] = probs_map[~zero_mask] / counter[~zero_mask]
        del counter

        logging.info('Total Network Run Time : {:.4f}'.format(time_total))
    return probs_map, time_total


def make_dataloader(args, cnn, slide, level_ckpt, assign, flip='NONE', rotate='NONE'):
    batch_size = cnn['batch_inf_size']
    num_workers = args.num_workers

    dataloader = DataLoader(
        WSIPatchDataset(slide, level_ckpt, assign,
                        image_size=cnn['patch_inf_size'],
                        normalize=True, flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    # configuration
    level_save = 3
    level_show = 6
    level_sample = int(args.probs_map_path.split('l')[-1])
    level_ckpt = int(args.ckpt_path.split('l')[-1])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f1:
        cnn = json.load(f1)
    ckpt = torch.load(os.path.join(args.ckpt_path, 'best.ckpt'))
    model = chose_model(cnn['model'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()
    
    with open(args.assign_path, 'r') as f2:
        assign = json.load(f2)

    time_total = 0.0
    dir = os.listdir(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l{}'.format(level_sample)))
    for file in sorted(dir):
        # if os.path.exists(os.path.join(args.probs_map_path, 'model_l{}'.format(level_ckpt), 'save_l{}'.format(level_save), file)):
        #     continue
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        tissue = np.load(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l{}'.format(level_sample), file))
                
        assign_per_img = assign[os.path.join(args.wsi_path, file.split('.')[0]+'.tif')]

        if len(assign_per_img) == 0:
            probs_map = np.zeros(tuple([int(i / 2**level_ckpt) for i in slide.level_dimensions[0]]))
        else:
            dataloader = make_dataloader(
                args, cnn, slide, level_ckpt, assign_per_img, flip='NONE', rotate='NONE')
            probs_map, time_network = get_probs_map(model, slide, level_ckpt, dataloader)
            time_total += time_network
        
        # save heatmap
        probs_map = (probs_map * 255).astype(np.uint8)
        shape_save = tuple([int(i / 2**level_save) for i in slide.level_dimensions[0]])
        probs_map = cv2.resize(probs_map, (shape_save[1], shape_save[0]), interpolation=cv2.INTER_CUBIC)
        np.save(os.path.join(args.probs_map_path, 'model_l{}'.format(level_ckpt), \
                                 'save_na_ns_l{}'.format(level_save), file.split('.')[0] + '.npy'), probs_map)

        # visulize heatmap
        img_rgb = slide.read_region((0, 0), level_show, \
                            tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        probs_map = cv2.resize(probs_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        probs_img_rgb = cv2.applyColorMap(probs_map, cv2.COLORMAP_JET)
        probs_img_rgb = cv2.cvtColor(probs_img_rgb, cv2.COLOR_BGR2RGB)
        heat_img = cv2.addWeighted(probs_img_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        cv2.imwrite(os.path.join(args.probs_map_path, 'model_l{}'.format(level_ckpt), \
                                   'save_na_ns_l{}'.format(level_save), file.split('.')[0] + '_heat.png'), heat_img)

    time_total_avg = time_total / len(dir)
    logging.info('AVG Total Run Time : {:.2f}'.format(time_total_avg))

def main():
    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/test/images",
        "/home/ps/hhy/slfcd/save_train/train_base_l1",
        "/home/ps/hhy/slfcd/camelyon16/configs/cnn_base_l1.json",
        '/media/ps/passport2/hhy/camelyon16/test/dens_map_select_l6',
        "/media/ps/passport2/hhy/camelyon16/test/crop_split_2048_sampling_l1/results.json"])
    args.GPU = "1"

    # args = parser.parse_args([
    #     "/media/hy/hhy_data/camelyon16/test/images",
    #     "/home/cka/hhy/slfcd/save_train/train_base_l1",
    #     "/home/cka/hhy/slfcd/camelyon16/configs/cnn_base_l1.json",
    #     '/media/hy/hhy_data/camelyon16/test/dens_map_assign_l6',
    #     "/media/hy/hhy_data/camelyon16/test/testset_assign_and_move.json"])
    # args.GPU = "2"
    run(args)


if __name__ == '__main__':
    main()