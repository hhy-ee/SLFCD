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
# from torchvision import models
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from camelyon16.data.prob_producer_bin import WSIPatchDataset  # noqa
from camelyon16 import models

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
parser.add_argument('--level', default=6, type=int, help='level for WSI, to '
                    'generate probs map, default 6')
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
    if mod == 'inception_v3':
        model = models.inception_v3(pretrained=False, aux_logits=False)
    elif mod== 'resnet18':
        model = models.resnet18(pretrained=False)
    else:
        raise Exception("I have not add any models. ")
    return model


def get_probs_map(model, size, dataloader):
    probs_map = np.zeros(size)
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    time_total = 0.
    with torch.no_grad():
        for (data, rect) in dataloader:
            data = Variable(data.cuda(non_blocking=True))
            output = model(data)
            # because of torch.squeeze at the end of forward in resnet.py, if the
            # len of dim_0 (batch_size) of data is 1, then output removes this dim.
            # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
            probs = output.sigmoid().cpu().data.numpy()
            for bs in range(probs.shape[0]):
                probs_map[rect[0][bs]:rect[2][bs], rect[1][bs]:rect[3][bs]] = probs[bs, :, :, 0]
            count += 1
            
            time_spent = time.time() - time_now
            time_now = time.time()
            logging.info(
                '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
                .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                    dataloader.dataset._rotate, count, num_batch, time_spent))
            time_total += time_spent
        logging.info('Total Run Time : {:.2f}'.format(time_total))
    return probs_map


def make_dataloader(args, cnn, slide, tissue, flip='NONE', rotate='NONE'):
    batch_size = cnn['batch_inf_size']
    num_workers = args.num_workers
    image_level = cnn['patch_inf_size']
    image_size = ((((image_level*16+1)*2+1)*2+3)*2+3)*2+1
    dataloader = DataLoader(
        WSIPatchDataset(slide, tissue, args.level, args.subdivisions,
                        image_level=image_level, image_size=image_size,
                        normalize=True,
                        flip=flip, 
                        rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f:
        cnn = json.load(f)
    # dir = os.listdir(args.wsi_path)
    dir = os.listdir('/media/ps/passport2/hhy/camelyon16/testing/tumor_mask/heat_image/')
    for file in dir:
        # slide = openslide.OpenSlide(os.path.join(args.wsi_path, file))
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        tissue = np.load(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask', file.split('.')[0]+'.npy'))
        ckpt = torch.load(args.ckpt_path)
        model = chose_model(cnn['model'])
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 1)
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda().eval()

        if not args.eight_avg:
            dataloader = make_dataloader(
                args, cnn, slide, tissue, flip='NONE', rotate='NONE')
            probs_map = get_probs_map(model, slide.level_dimensions[args.level], dataloader)
        else:
            probs_map = np.zeros(slide.level_dimensions[args.level])

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

        # np.save(os.path.join(args.probs_map_path, file.split('.')[0] + '.npy'), probs_map)
        probs_map_img = np.asarray(probs_map * 255, dtype=np.uint8)
        probs_map_img = cv2.applyColorMap(probs_map_img, cv2.COLORMAP_JET)
        probs_map_img = cv2.cvtColor(probs_map_img, cv2.COLOR_BGR2RGB)
        probs_map_img = Image.fromarray(probs_map_img.transpose((1,0,2)))
        heat_img = Image.blend(dataloader.dataset._img, probs_map_img, 0.3)
        probs_map_img.save(os.path.join(args.probs_map_path, file.split('.')[0] + '.png'))
        heat_img.save(os.path.join(args.probs_map_path, file.split('.')[0] + '_heat.png'))



def main():
    args = parser.parse_args([
        "/media/ps/passport2/hhy/camelyon16/testing/images",
        "/home/ps/hhy/slfcd/save_train/train_bin/best.ckpt",
        "/home/ps/hhy/slfcd/camelyon16/configs/cnn_bin.json",
        '/media/ps/passport2/hhy/camelyon16/testing/probs_map_bin/'])
    args.level = 3
    args.subdivisions = 2
    run(args)


if __name__ == '__main__':
    main()
