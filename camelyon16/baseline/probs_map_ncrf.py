import sys
import os
import argparse
import logging
import json
import time
import cv2
from skimage import transform

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

from camelyon16.data.prob_producer_ncrf import WSIPatchDataset  # noqa
from camelyon16.models.ncrf_model import MODELS


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
parser.add_argument('--GPU', default='1', type=str, help='which GPU to use'
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


def get_probs_map(model, dataloader):

    probs_map = np.zeros(dataloader.dataset._tissue.shape)
    num_batch = len(dataloader)
    idx_center = dataloader.dataset._grid_size // 2
    
    count = 0
    time_now = time.time()
    time_total = 0.
    with torch.no_grad():
        for (data, x_mask, y_mask) in dataloader:
            data = Variable(data.cuda(non_blocking=True), volatile=True)
            output = model(data)
            # because of torch.squeeze at the end of forward in resnet.py, if the
            # len of dim_0 (batch_size) of data is 1, then output removes this dim.
            # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
            if len(output.shape) == 1:
                probs = output[idx_center].sigmoid().cpu().data.numpy().flatten()
            else:
                probs = output[:, idx_center].sigmoid().cpu().data.numpy().flatten()
            
            probs_map[x_mask, y_mask] = probs
            count += 1
            
            time_spent = time.time() - time_now
            time_now = time.time()
            logging.info(
                '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
                .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                    dataloader.dataset._rotate, count, num_batch, time_spent))
            time_total += time_spent
        logging.info('Total Network Run Time : {:.4f}'.format(time_total))
    return probs_map, time_total


def make_dataloader(args, cnn, slide, tissue, flip='NONE', rotate='NONE'):
    batch_size = cnn['batch_size']
    num_workers = args.num_workers
    
    dataloader = DataLoader(
        WSIPatchDataset(slide, tissue, image_size=cnn['image_size'],
                        normalize=True, flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
     # configuration
    level_show = 6
    level = int(args.probs_map_path.split('l')[-1])
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f:
        cnn = json.load(f)
    ckpt = torch.load(args.ckpt_path)
    model = MODELS['resnet18'](num_nodes=(cnn['image_size']//256)**2, use_crf='ncrf' in args.probs_map_path)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()

    time_total = 0.0
    dir = os.listdir(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l{}'.format(level)))
    for file in sorted(dir):
        # if os.path.exists(os.path.join(args.probs_map_path, file)):
        #     continue
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        tissue = np.load(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l{}'.format(level), file))
        
        # calculate heatmap & runtime
        dataloader = make_dataloader(
            args, cnn, slide, tissue, flip='NONE', rotate='NONE')
        probs_map, time_network = get_probs_map(model, dataloader)
        time_total += time_network

        # save heatmap
        np.save(os.path.join(args.probs_map_path, file.split('.')[0] + '.npy'), probs_map)
        probs_map = (probs_map * 255).astype(np.uint8)
        
        # visulize heatmap
        img_rgb = slide.read_region((0, 0), level_show, \
                            tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        probs_map = cv2.resize(probs_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        probs_img_rgb = cv2.applyColorMap(probs_map, cv2.COLORMAP_JET)
        probs_img_rgb = cv2.cvtColor(probs_img_rgb, cv2.COLOR_BGR2RGB)
        heat_img = cv2.addWeighted(probs_img_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        cv2.imwrite(os.path.join(args.probs_map_path, file.split('.')[0] + '_heat.png'), heat_img)

    time_total_avg = time_total / len(dir)
    logging.info('AVG Total Run Time : {:.2f}'.format(time_total_avg))

def main():  
    args = parser.parse_args([
        "./datasets/test/images",
        "./save_train/train_ncrf/resnet18_ncrf.ckpt",
        "./camelyon16/configs/cnn_ncrf.json",
        './datasets/test/dens_map_ncrf_l6'])
    args.GPU = "3"
    
    run(args)


if __name__ == '__main__':
    main()
