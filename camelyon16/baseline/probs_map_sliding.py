import sys
import os
import argparse
import logging
import json
import time
import cv2
from PIL import Image

import numpy as np
import torch
import openslide
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from camelyon16.data.prob_producer_base_sliding import WSIPatchDataset  # noqa


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
parser.add_argument('--GPU', default='2', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=0, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--overlap', default=0, type=int, help='whether to'
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
    denominator = np.zeros(tuple([int(i / 2**level) for i in slide.level_dimensions[0]]))
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    time_total = 0.
    with torch.no_grad():
        for (data, rect, shape, keep) in dataloader:
            data = Variable(data[keep].cuda(non_blocking=True))
            count += 1
            if data.shape[0] == 0:
                continue
            output = model(data)
            # because of torch.squeeze at the end of forward in resnet.py, if the
            # len of dim_0 (batch_size) of data is 1, then output removes this dim.
            # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
            probs = output['out'][:, :].sigmoid().cpu().data.numpy()
            for bs in range(probs.shape[0]):
                denominator[rect[0][keep][bs]:rect[2][keep][bs], rect[1][keep][bs]:rect[3][keep][bs]] += 1
                probs_map[rect[0][keep][bs]:rect[2][keep][bs], rect[1][keep][bs]:rect[3][keep][bs]] += \
                    probs[bs, 0, :shape[0][keep][bs], :shape[1][keep][bs]]
            
            time_spent = time.time() - time_now
            time_now = time.time()
            logging.info(
                '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
                .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                    dataloader.dataset._rotate, count, num_batch, time_spent))
            time_total += time_spent
        denominator = denominator + (denominator < 1)*1
        probs_map = probs_map / denominator
        logging.info('Total Network Run Time : {:.4f}'.format(time_total))
    return probs_map, time_total


def make_dataloader(args, cnn, slide, tissue, level, flip='NONE', rotate='NONE'):
    batch_size = cnn['batch_inf_size']
    num_workers = args.num_workers
    
    dataloader = DataLoader(
        WSIPatchDataset(slide, tissue, level, args.overlap, 
                        image_size=cnn['patch_inf_size'],
                        normalize=True, flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f:
        cnn = json.load(f)
    dir = os.listdir(args.wsi_path)
    level = int(args.probs_map_path.split('l')[-1])
    time_total = 0.0
    for file in dir:
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        tissue = np.load(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l6', file.split('.')[0]+'.npy'))
        ckpt = torch.load(args.ckpt_path)
        model = chose_model(cnn['model'])
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda().eval()

        if not args.eight_avg:
            dataloader = make_dataloader(
                args, cnn, slide, tissue, level, flip='NONE', rotate='NONE')
            probs_map, time_network = get_probs_map(model, slide, level, dataloader)
            time_total += time_network
        else:
            probs_map = np.zeros(slide.level_dimensions[level])

            dataloader = make_dataloader(
                args, cnn, flip='NONE', rotate='NONE')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cnn, flip='NONE', rotate='ROTATE_90')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cnn, flip='NONE', rotate='ROTATE_180')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cnn, flip='NONE', rotate='ROTATE_270')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cnn, flip='FLIP_LEFT_RIGHT', rotate='NONE')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
            probs_map += get_probs_map(model, dataloader)

            probs_map /= 8

        tissue_mask = Image.fromarray(tissue.transpose()).resize(probs_map.shape)
        probs_map = probs_map * np.asarray(tissue_mask).transpose()
        probs_map = cv2.GaussianBlur((probs_map * 255).astype(np.uint8), (13,13), 11)
        np.save(os.path.join(args.probs_map_path, file.split('.')[0] + '.npy'), probs_map)

        level_show = 4
        img_rgb = slide.read_region((0, 0), level_show, tuple([int(i / 2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        probs_img_rgb = Image.fromarray(probs_map.transpose()).resize(img_rgb.shape[:2])
        probs_img_rgb= cv2.applyColorMap(np.asarray(probs_img_rgb).transpose(), cv2.COLORMAP_JET)
        probs_img_rgb = cv2.cvtColor(probs_img_rgb, cv2.COLOR_BGR2RGB)
        heat_img = cv2.addWeighted(probs_img_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        cv2.imwrite(os.path.join(args.probs_map_path, file.split('.')[0] + '_heat.png'), heat_img)


    time_total_avg = time_total / len(dir)
    logging.info('AVG Total Run Time : {:.2f}'.format(time_total_avg))

def main():
    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/train/tumor",
        "/home/ps/hhy/slfcd/save_train/train_base_l2/best.ckpt",
        "/home/ps/hhy/slfcd/camelyon16/configs/cnn_base_l2.json",
        '/media/ps/passport2/hhy/camelyon16/train/dens_map_sliding_l2'])
    args.overlap = 0
    args.GPU = "1"

    # args = parser.parse_args([
    #     "/media/ps/passport2/hhy/camelyon16/test/images",
    #     "/home/ps/hhy/slfcd/save_train/train_base_l3/best.ckpt",
    #     "/home/ps/hhy/slfcd/camelyon16/configs/cnn_base_l3.json",
    #     '/media/ps/passport2/hhy/camelyon16/test/dens_map_sliding_l3'])
    run(args)


if __name__ == '__main__':
    main()
