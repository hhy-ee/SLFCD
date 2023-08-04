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
parser.add_argument('--itc_threshold', default=[100,500], metavar='ITC_THRESHOLD',
                    type=float, help='threshold of the long axis of isolated tumor')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
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
        probs_map = prior / 255
        counter = np.ones(shape)
    else:
        probs_map = np.zeros(shape)
        counter = np.zeros(shape)

    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    time_total = 0.
    
    with torch.no_grad():
        for (data, rect, box) in dataloader:
            data = Variable(data.cuda(non_blocking=True))
            output = model(data)
            probs = output['out'][:, :].sigmoid().cpu().data.numpy()

            rect = [(item / resolution).to(torch.int) for item in rect]
            box = [(item / resolution).to(torch.int) for item in box]
            for bs in range(probs.shape[0]):
                left, top, right, bot = rect[0][bs], rect[1][bs], rect[2][bs], rect[3][bs]
                l, t, r, b, s = box[0][bs], box[1][bs], box[2][bs], box[3][bs], box[4][bs]
                prob = transform.resize(probs[bs, 0], (s, s))
                counter[left: right, top: bot] += 1
                probs_map[left: right, top: bot] += prob[l: r, t: b]

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


def make_dataloader(args, file, cnn, slide, prior, level_sample, level_ckpt, flip='NONE', rotate='NONE'):
    batch_size = cnn['batch_inf_size']
    num_workers = args.num_workers

    dataloader = DataLoader(
        WSIPatchDataset(slide, prior, level_sample, level_ckpt, args, file,
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
    patch_total = 0
    dir = os.listdir(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l6'))
    for file in sorted(dir)[:40]:
        # if os.path.exists(os.path.join(args.probs_path, 'model_{}_l{}'.format(args.roi_generator, level_ckpt), \
        #                             'save_roi_th_0.1_min1e0_max1e9_whole_fixmodel_l{}'.format(level_save), file)):
        #     continue
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        
        # compute Point of Interest (POI)
        first_stage_map = np.load(os.path.join(args.prior_path, file))
        shape = tuple([int(i / 2**level_sample) for i in slide.level_dimensions[0]])
        prior_map = cv2.resize(first_stage_map, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        POI = (prior_map / 255) > args.roi_threshold
        # Computes the inference mask
        filled_image = nd.morphology.binary_fill_holes(POI)
        evaluation_mask = measure.label(filled_image, connectivity=2)
        # eliminate large TC
        max_label = np.amax(evaluation_mask)
        properties = measure.regionprops(evaluation_mask)
        filled_mask = np.zeros(filled_image.shape) > 0
        feature_map = np.zeros(filled_image.shape).astype(np.uint8)
        threshold = tuple([t / (0.243 * pow(2, level_sample)) for t in args.itc_threshold])
        for i in range(0, max_label):
            if properties[i].major_axis_length > threshold[0] and properties[i].major_axis_length < threshold[1]:
                l, t, r, b = properties[i].bbox
                filled_mask[l: r, t: b] = np.logical_or(filled_mask[l: r, t: b], properties[i].image_filled)
                region_confidence = first_stage_map[properties[i].coords[:,0], properties[i].coords[:,1]].mean()
                feature_map[properties[i].coords[:,0], properties[i].coords[:,1]] = region_confidence
        
        # generate distance map
        distance, coord = nd.distance_transform_edt(filled_mask, return_indices=True)
        prior = (prior_map, distance, coord, feature_map)

        # calculate heatmap & runtime
        dataloader = make_dataloader(
            args, file, cnn, slide, prior, level_sample, level_ckpt, flip='NONE', rotate='NONE')
        probs_map, time_network = get_probs_map(model, slide, level_save, level_ckpt, dataloader, prior=first_stage_map)
        patch_total += dataloader.dataset._idcs_num
        time_total += time_network

        # save heatmap
        probs_map = (probs_map * 255).astype(np.uint8)
        shape_save = tuple([int(i / 2**level_save) for i in slide.level_dimensions[0]])
        probs_map = cv2.resize(probs_map, (shape_save[1], shape_save[0]), interpolation=cv2.INTER_CUBIC)
        np.save(os.path.join(args.probs_path, 'model_{}_l{}'.format(args.roi_generator, level_ckpt), \
            'save_roi_th_0.01_min1e0_max5e2_edge_fixmodel_fixsize1x256_l{}'.format(level_save), file), probs_map)

        # visulize heatmap
        img_rgb = slide.read_region((0, 0), level_show, \
                            tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        probs_map = cv2.resize(probs_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        probs_img_rgb = cv2.applyColorMap(probs_map, cv2.COLORMAP_JET)
        probs_img_rgb = cv2.cvtColor(probs_img_rgb, cv2.COLOR_BGR2RGB)
        heat_img = cv2.addWeighted(probs_img_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        cv2.imwrite(os.path.join(args.probs_path, 'model_{}_l{}'.format(args.roi_generator, level_ckpt), \
            'save_roi_th_0.01_min1e0_max5e2_edge_fixmodel_fixsize1x256_l{}'.format(level_save), file.split('.')[0] + '_heat.png'), heat_img)

    time_total_avg = time_total / len(dir)
    logging.info('AVG Run Time : {:.2f}'.format(time_total_avg))
    logging.info('Total Patch Number : {:d}'.format(patch_total))
    logging.info('AVG Patch Number : {:2f}'.format(patch_total / len(dir)))
    
def main():
    args = parser.parse_args([
        "./datasets/test/images",
        "./save_train/train_fix_nobg_l1",
        "./camelyon16/configs/cnn_fix_l1.json",
        './datasets/test/dens_map_sampling1_l8/model_l1/save_l3',
        './datasets/test/dens_map_sampling_2s_l6'])
    args.roi_generator = 'prior_l8'
    args.roi_threshold = 0.01
    args.itc_threshold = [1e0, 5e2]
    args.GPU = "2"
    
    run(args)


if __name__ == '__main__':
    main()