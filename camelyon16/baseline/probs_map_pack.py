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

from camelyon16.data.prob_producer_pack import WSIPatchDataset  # noqa


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
parser.add_argument('--canvas_size', default=800, metavar='CANVAS_SIZE',
                    type=float, help='The size of the canvas to pack the patches')
parser.add_argument('--patch_size', default=256, metavar='PATCH_SIZE',
                    type=float, help='The size of the canvas to pack the patches')
parser.add_argument('--fill_in', default=False, help='whether to resize'
                    'the patch to fill in the blank region of the canvas')
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


def get_probs_map(model, slide, level_ckpt, dataloader, prior=None, level_prior=3):

    shape = tuple([int(i / 2**level_prior) for i in slide.level_dimensions[0]])
    resolution = 2 ** (level_prior - level_ckpt)
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
        for (data, box, canvas, patch) in dataloader:
            data = Variable(data.cuda(non_blocking=True))
            output = model(data)
            probs = output['out'][:, :].sigmoid().cpu().data.numpy()
            
            box = [[(item / resolution).to(torch.int) for item in list] for list in box]
            patch = [[(item / resolution).to(torch.int) for item in list] for list in patch]
            for bs in range(len(probs)):
                for pt in range(len(box)):
                    b_l, b_t, b_r, b_b = box[pt][0][bs], box[pt][1][bs], box[pt][2][bs], box[pt][3][bs]
                    c_l, c_t, c_r, c_b = canvas[pt][0][bs], canvas[pt][1][bs], canvas[pt][2][bs], canvas[pt][3][bs]
                    p_l, p_t, p_r, p_b, p_x, p_y = patch[pt][0][bs], patch[pt][1][bs], patch[pt][2][bs], \
                                                                patch[pt][3][bs], patch[pt][4][bs], patch[pt][5][bs]
                    prob = transform.resize(probs[bs, 0, c_l: c_r, c_t: c_b], (p_x, p_y))
                    counter[b_l: b_r, b_t: b_b] += 1
                    probs_map[b_l: b_r, b_t: b_b] += prob[p_l: p_r, p_t: p_b]
                    
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
    batch_size = cnn['batch_size']
    num_workers = args.num_workers

    dataloader = DataLoader(
        WSIPatchDataset(slide, prior, level_sample, level_ckpt, args, file,
                        image_size=args.patch_size, normalize=True, flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    # configuration
    level_save = 6
    level_show = 6
    level_sample = int(args.probs_path.split('l')[-1])
    level_ckpt = int(args.ckpt_path.split('l')[-1])
    overlap = os.path.basename(args.prior_path).split('_')[-2].split('o')[-1]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)
    
    with open(args.assign_path, 'r') as f_assign:
        assign = json.load(f_assign)
        
    with open(os.path.join(os.path.dirname(args.assign_path), 'results.json'), 'r') as f_cluster:
        cluster = json.load(f_cluster)
        
    info = assign['data_info']
    save_path = os.path.join(args.probs_path, 'model_prior_o{}_l{}'.format(overlap, level_ckpt), \
        'save_pack_roi_th_{}_itc_th_{}_canvas_{}_{}_patch_{}_shuffle_{}_pack_{}_{}_region_{}_model_{}_size_l{}'.format(\
        info['th_roi'], info['th_itc'], args.canvas_size, args.patch_type, args.patch_size, args.random_shuffle, \
        args.pack_mode, info['sample_type'], os.path.basename(args.cnn_path).split('_')[1], info['patch_type'], level_ckpt))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    with open(args.cnn_path) as f:
        cnn = json.load(f)
    ckpt = torch.load(os.path.join(args.ckpt_path, 'best.ckpt'))
    model = chose_model(cnn['model'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()

    time_total = 0.0
    patch_total = 0
    canvas_total = 0
    dir = os.listdir(os.path.join(os.path.dirname(args.wsi_path), 'tissue_mask_l6'))
    for file in sorted(dir)[:40]:
        # if os.path.exists(os.path.join(save_path, file)):
        #     continue
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))
        first_stage_map = np.load(os.path.join(args.prior_path, file))
        shape = tuple([int(i / 2**level_sample) for i in slide.level_dimensions[0]])
        prior_map = cv2.resize(first_stage_map, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Get patches from assignment files
        file_keys = [key for key in assign.keys() if file.split('.')[0] in key]
        assign_per_img = [box for key in file_keys for box in assign[key]]
        parent_per_img = [{'keep': box['cluster']} for key in file_keys for box in cluster[key][1:]]
        child_per_img = [{'keep': child['cluster']} for key in file_keys for box in cluster[key][1:] for child in box['child']]
        if args.patch_type == 'child':
            cluster_per_img = child_per_img
        elif args.patch_type == 'parent':
            cluster_per_img = parent_per_img
        elif args.patch_type == 'total':
            cluster_per_img = parent_per_img + child_per_img
        
        # generate prior
        prior = (prior_map, cluster_per_img)
        
        # calculate heatmap & runtime
        dataloader = make_dataloader(
            args, file, cnn, slide, prior, level_sample, level_ckpt, flip='NONE', rotate='NONE')
        patch_total += len(dataloader.dataset._X_idcs)
        canvas_total += dataloader.dataset._idcs_num
        probs_map, time_network = get_probs_map(model, slide, level_ckpt, dataloader, prior=first_stage_map)
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
    logging.info('Total Canvas Number : {:d}'.format(canvas_total))
    logging.info('AVG Patch Number : {:.2f}'.format(patch_total / len(dir)))
    logging.info('AVG Canvas Number : {:.2f}'.format(canvas_total / len(dir)))
    
def main():
    args = parser.parse_args([
        "./datasets/test/images",
        "./save_train/train_fix_l1",
        "./camelyon16/configs/cnn_fix_l1.json",
        './datasets/test/prior_map_sampling_o0.5_l1',
        './datasets/test/dens_map_sampling_2s_l6'])
    args.canvas_size = 800
    args.patch_size = 256
    args.patch_type = 'total'
    args.pack_mode = 'crop'
    args.random_shuffle = False
    args.GPU = "2"
    
    args.assign_path = "./datasets/test/patch_cluster_l1/cluster_roi_th_0.1_itc_th_1e0_1e9_nms_1.0_nmm_0.5_whole_region_fix_size_l1/results_boxes.json"
    run(args)


if __name__ == '__main__':
    main()