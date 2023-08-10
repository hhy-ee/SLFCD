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

from camelyon16.data.prob_producer_select_t_assign_t import WSIPatchDataset  # noqa


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file related to the ckpt file')
parser.add_argument('prior_path', default=None, metavar='PRIOR_MAP_PATH',
                    type=str, help='Path to the result of first stage numpy file')
parser.add_argument('probs_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--assign_path', default=None, metavar='ASSIGN_PATH',
                    help='Path to the result of assignment numpy file')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use')
parser.add_argument('--num_workers', default=0, type=int, help='number of '
                    'workers to use to make batch, default 5')
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
        for (data, box, moved_box) in dataloader:
            data = Variable(data.cuda(non_blocking=True))
            output = model(data)
            probs = output['out'][:, :].sigmoid().cpu().data.numpy()
            box = box // resolution
            
            for i in range(probs.shape[0]):
                for j in range(box.shape[1]):
                    counter[box[i][j][0]:box[i][j][2], box[i][j][1]:box[i][j][3]] += 1
                    patch_prob = probs[i, 0, moved_box[i][j][0]:moved_box[i][j][2], moved_box[i][j][1]:moved_box[i][j][3]]
                    if patch_prob.shape != (0,0):
                        patch_prob = transform.resize(patch_prob, (box[i][j][2]-box[i][j][0], box[i][j][3]-box[i][j][1]))
                        probs_map[box[i][j][0]:box[i][j][2], box[i][j][1]:box[i][j][3]] += patch_prob
            
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
    batch_size = cnn['batch_size']
    num_workers = args.num_workers

    dataloader = DataLoader(
        WSIPatchDataset(slide, level_ckpt, assign,
                        image_size=cnn['image_size'],
                        normalize=True, flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    # configuration
    level_save = 3
    level_show = 6
    level_sample = int(args.probs_path.split('l')[-1])
    level_ckpt = int(args.ckpt_path.split('l')[-1])
    overlap = os.path.basename(args.prior_path).split('_')[-2].split('o')[-1]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)
    
    save_path = os.path.join(args.probs_path,  'model_prior_o{}_l{}'.format(overlap, level_ckpt), \
                'save_roi_th_{}_itc_th_{}_canvas_{}_patch_{}_{}_fixmodel_dynsize_l{}'.format(args.roi_threshold, \
                args.itc_threshold, args.canvas_size, args.patch_size, args.sample_type, level_save))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    with open(args.cnn_path) as f:
        cnn = json.load(f)
    ckpt = torch.load(os.path.join(args.ckpt_path, 'best.ckpt'))
    model = chose_model(cnn['model'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()
    
    with open(args.assign_path, 'r') as f_assign:
        assign = json.load(f_assign)

    time_total = 0.0
    patch_total = 0
    dir = os.listdir(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l6'))
    for file in sorted(dir)[:40]:
        # if os.path.exists(os.path.join(args.probs_path, 'model_prior_o{}_l{}'.format(overlap, level_ckpt), \
        #           'save_roi_th_0.01_itc_th_1e0_5e2_edge_fixmodel_fixsize1x256_l{}'.format(level_save), file)):
        #     continue
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        first_stage_map = np.load(os.path.join(args.prior_path, file))
        shape = tuple([int(i / 2**level_sample) for i in slide.level_dimensions[0]])
        prior_map = cv2.resize(first_stage_map, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Get patches from assignment files
        assign_per_img = []
        for item in assign:
            if item['file_name'] == file.split('.')[0]:
                assign_per_img.append(item)
        # calculate heatmap & runtime
        dataloader = make_dataloader(
            args, cnn, slide, level_ckpt, assign_per_img, flip='NONE', rotate='NONE')
        probs_map, time_network = get_probs_map(model, slide, level_save, level_ckpt, dataloader, prior=first_stage_map)
        patch_total += dataloader.dataset._idcs_num
        time_total += time_network
        
        # save heatmap
        probs_map = (probs_map * 255).astype(np.uint8)
        shape_save = tuple([int(i / 2**level_save) for i in slide.level_dimensions[0]])
        probs_map = cv2.resize(probs_map, (shape_save[1], shape_save[0]), interpolation=cv2.INTER_CUBIC)
        np.save(os.path.join(save_path, file), probs_map)

        # visulize heatmap
        img_rgb = slide.read_region((0, 0), level_show, \
                            tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        probs_map = cv2.resize(probs_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        probs_img_rgb = cv2.applyColorMap(probs_map, cv2.COLORMAP_JET)
        probs_img_rgb = cv2.cvtColor(probs_img_rgb, cv2.COLOR_BGR2RGB)
        heat_img = cv2.addWeighted(probs_img_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        cv2.imwrite(os.path.join(save_path, file.split('.')[0] + '_heat.png'), heat_img)

    time_total_avg = time_total / len(dir)
    logging.info('AVG Run Time : {:.2f}'.format(time_total_avg))
    logging.info('Total Patch Number : {:d}'.format(patch_total))
    logging.info('AVG Patch Number : {:.2f}'.format(patch_total / len(dir)))
    
def main():
    args = parser.parse_args([
        "./datasets/test/images",
        "./save_train/train_fix_l1",
        "./camelyon16/configs/cnn_fix_l1.json",
        './datasets/test/prior_map_sampling_o0.25_l1',
        './datasets/test/dens_map_sampling_2s_l6'])
    args.canvas_size = 800
    args.patch_size = 256
    args.GPU = "2"
    
    args.assign_path = "./datasets/test/crop_split_l1/assign.json",
    run(args)


if __name__ == '__main__':
    main()
    
    # if len(assign_per_img) == 0:
    #     probs_map = np.zeros(tuple([int(i / 2**level_save) for i in slide.level_dimensions[0]]))
    # else: