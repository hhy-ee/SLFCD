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
from scipy import ndimage as nd
from skimage import measure
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from camelyon16.data.prob_producer_base_random_sampling import WSIPatchDataset  # noqa


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('prior_path', default=None, metavar='PRIOR_MAP_PATH',
                    type=str, help='Path to the result of first stage numpy file')
parser.add_argument('probs_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--roi_generator', default='sampling_l8', metavar='ROI_GENERATOR',
                    type=str, help='type of the generator of the first stage')
parser.add_argument('--roi_threshold', default=0.1, metavar='ROI_GENERATOR',
                    type=float, help='threshold of the generator of the first stage')
parser.add_argument('--itc_threshold', default=100, metavar='ITC_THRESHOLD',
                    type=float, help='threshold of the long axis of isolated tumor')
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


def get_probs_map(model, slide, level_save, level_ckpt, dataloader, prior=None):

    shape = tuple([int(i / 2**level_save) for i in slide.level_dimensions[0]])
    resolution = 2 ** (level_save - level_ckpt)
    if prior is not None:
        probs_map = cv2.resize(prior, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) / 255
        counter = np.ones(shape)
    else:
        probs_map = np.zeros(shape)
        counter = np.zeros(shape)
    
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    time_total = 0.
    with torch.no_grad():
        for (data, rect, shape) in dataloader:
            data = Variable(data.cuda(non_blocking=True))
            output = model(data)
            probs = output['out'][:, :].sigmoid().cpu().data.numpy()

            rect = [(item / resolution).to(torch.int) for item in rect]
            for bs in range(probs.shape[0]):
                l, t, r, b = rect[0][bs], rect[1][bs], rect[2][bs], rect[3][bs]
                prob = transform.resize(probs[bs, 0], (r - l, b - t))
                counter[l: r, t: b] += 1
                probs_map[l: r, t: b] += prob

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


def make_dataloader(args, cnn, slide, prior, level_sample, level_ckpt, flip='NONE', rotate='NONE'):
    batch_size = cnn['batch_inf_size']
    num_workers = args.num_workers
    
    dataloader = DataLoader(
        WSIPatchDataset(slide, prior, level_sample, level_ckpt, 
                        image_size=cnn['patch_inf_size'],
                        normalize=True, flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    # configuration
    level_save = 3
    level_show = 6
    level_sample = int(args.probs_path.split('l')[-1])
    level_ckpt = int(args.ckpt_path.split('l')[-1])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)
        
    with open(args.cnn_path) as f:
        cnn = json.load(f)
    ckpt = torch.load(os.path.join(args.ckpt_path, 'best.ckpt'))
    model = chose_model(cnn['model'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()
    
    time_total = 0.0
    dir = os.listdir(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l{}'.format(level_sample)))
    for file in sorted(dir)[111:128]:
        # if os.path.exists(os.path.join(args.probs_path, 'model_{}_l{}'.format(args.roi_generator, level_ckpt), \
        #                     'save_random_l{}'.format(level_save), file)):
        #     continue
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        
        # compute Point of Interest (POI)
        first_stage_map = np.load(os.path.join(args.prior_path, file))
        shape = tuple([int(i / 2**level_sample) for i in slide.level_dimensions[0]])
        first_stage_map = cv2.resize(first_stage_map, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        POI = (first_stage_map / 255) > args.roi_threshold
        # Computes the inference mask
        filled_image = nd.morphology.binary_fill_holes(POI)
        evaluation_mask = measure.label(filled_image, connectivity=2)
        # eliminate ITC
        max_label = np.amax(evaluation_mask)
        properties = measure.regionprops(evaluation_mask)
        filled_mask = np.zeros(first_stage_map.shape) > 0
        threshold = args.itc_threshold / (0.243 * pow(2, level_sample))
        for i in range(0, max_label):
            if properties[i].major_axis_length > threshold:
                l, t, r, b = properties[i].bbox
                filled_mask[l: r, t: b] = np.logical_or(filled_mask[l: r, t: b], properties[i].image_filled)
        # generate distance map
        distance, coord = nd.distance_transform_edt(filled_mask, return_indices=True)
        prior = (first_stage_map, distance, coord)

        # calculate heatmap & runtime
        dataloader = make_dataloader(
            args, cnn, slide, prior, level_sample, level_ckpt, flip='NONE', rotate='NONE')
        probs_map, time_network = get_probs_map(model, slide, level_save, level_ckpt, dataloader, prior=first_stage_map)
        time_total += time_network

        # save heatmap
        probs_map = (probs_map * 255).astype(np.uint8)
        shape_save = tuple([int(i / 2**level_save) for i in slide.level_dimensions[0]])
        probs_map = cv2.resize(probs_map, (shape_save[1], shape_save[0]), interpolation=cv2.INTER_CUBIC)
        np.save(os.path.join(args.probs_path, 'model_{}_l{}'.format(args.roi_generator, level_ckpt), \
                                    'save_roitt_0.5_min_100_edge_1_fix_size_256_non_holes_l{}'.format(level_save), file.split('.')[0] + '.npy'), probs_map)

        # visulize heatmap
        img_rgb = slide.read_region((0, 0), level_show, \
                            tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        probs_map = cv2.resize(probs_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        probs_img_rgb = cv2.applyColorMap(probs_map, cv2.COLORMAP_JET)
        probs_img_rgb = cv2.cvtColor(probs_img_rgb, cv2.COLOR_BGR2RGB)
        heat_img = cv2.addWeighted(probs_img_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        cv2.imwrite(os.path.join(args.probs_path, 'model_{}_l{}'.format(args.roi_generator, level_ckpt), \
                                    'save_roitt_0.5_min_100_edge_1_fix_size_256_non_holes_l{}'.format(level_save), file.split('.')[0] + '_heat.png'), heat_img)

    time_total_avg = time_total / len(dir)
    logging.info('AVG Total Run Time : {:.2f}'.format(time_total_avg))

def main():
    args = parser.parse_args([
        "./datasets/test/images",
        "./save_train/train_base_l1",
        "./camelyon16/configs/cnn_base_l1.json",
        './datasets/test/dens_map_sampling_l8/model_l1/save_l3',
        './datasets/test/dens_map_sampling_2s_l6'])
    args.roi_generator = 'distance'
    args.roi_threshold = 0.5
    args.GPU = "0"
    
    run(args)


if __name__ == '__main__':
    main()
