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

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from camelyon16.data.prob_producer_base_assign import WSIPatchDataset  # noqa


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
parser.add_argument('assign_path', default=None, metavar='ASSIGN_PATH',
                    help='Path to the result of assignment numpy file')
parser.add_argument('probs_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
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
        probs_map = cv2.resize(prior, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) / 255
        counter = np.ones(shape)
    else:
        probs_map = np.zeros(shape)
        counter = np.zeros(shape)

    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    time_total = 0.
    
    # file_name = slide._filename.split('/')[-1].replace('tif', 'npy')
    # if file_name in os.listdir('./datasets/test/tumor_mask_l6'):
    #     label = np.load(os.path.join('./datasets/test/tumor_mask_l6', file_name))
    #     label = transform.resize(label, (slide.level_dimensions[level_save]))
    # else:
    #     label = np.zeros(slide.level_dimensions[level_save])
        
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

            # #plot
            # probs_split = np.zeros((1024, 1024))
            # probs_aggre = np.zeros((1024, 1024))
            # label_canvas = np.zeros((1024, 1024))
            # for i in range(probs.shape[0]):
            #     for j in range(box.shape[1]):
            #         moved_box_w = int(moved_box[i][j][2]-moved_box[i][j][0])
            #         moved_box_h = int(moved_box[i][j][3]-moved_box[i][j][1])
            #         if moved_box_w != 0 and moved_box_h != 0:
            #             patch = data[:, :, moved_box[i][j][0]:moved_box[i][j][2], moved_box[i][j][1]:moved_box[i][j][3]]
            #             patch_resize = transform.resize(patch.cpu().data.numpy(), (1, 3, 256, 256))
            #             patch_resize = Variable(torch.tensor(patch_resize).cuda(non_blocking=True))
            #             output_resize = model(patch_resize)['out'][:, :].sigmoid().cpu().data.numpy()
            #             output_patch1 = transform.resize(output_resize[0,0,:], (moved_box_w, moved_box_h))
            #             output_patch2 = probs[i, 0, moved_box[i][j][0]:moved_box[i][j][2], moved_box[i][j][1]:moved_box[i][j][3]]
            #             probs_split[moved_box[i][j][0]:moved_box[i][j][2], moved_box[i][j][1]:moved_box[i][j][3]] = output_patch1
            #             probs_aggre[moved_box[i][j][0]:moved_box[i][j][2], moved_box[i][j][1]:moved_box[i][j][3]] = output_patch2
            #             label_patch = label[box[i][j][0]:box[i][j][2], box[i][j][1]:box[i][j][3]]
            #             label_patch = transform.resize(label_patch, (moved_box_w, moved_box_h))
            #             label_canvas[moved_box[i][j][0]:moved_box[i][j][2], moved_box[i][j][1]:moved_box[i][j][3]] = label_patch
                        
            # cv2.imwrite('./datasets/test/dens_map_assign_l6/example/{}_bg.png'.format(count), \
            #                         ((data.cpu().data.numpy()*128)+128).astype(np.uint8)[0,:].transpose(1,2,0))
            # cv2.imwrite('./datasets/test/dens_map_assign_l6/example/{}_pd1.png'.format(count), \
            #                         (probs_aggre*255).astype(np.uint8))
            # cv2.imwrite('./datasets/test/dens_map_assign_l6/example/{}_pd2.png'.format(count), \
            #                         (probs_split*255).astype(np.uint8))
            # cv2.imwrite('./datasets/test/dens_map_assign_l6/example/{}_gt.png'.format(count), \
            #                         (label_canvas*255).astype(np.uint8))
            
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
    for file in sorted(dir)[80:]:
        # if os.path.exists(os.path.join(args.probs_map_path, 'model_l{}'.format(level_save), 'save_l{}'.format(level_save), file)):
        #     continue
        slide = openslide.OpenSlide(os.path.join(args.wsi_path, file.split('.')[0]+'.tif'))

        first_stage_map = np.load(os.path.join(args.prior_path, file))
        shape = tuple([int(i / 2**level_sample) for i in slide.level_dimensions[0]])
        first_stage_map = cv2.resize(first_stage_map, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        
        assign_per_img = []
        for item in assign:
            if item['file_name'] == file.split('.')[0]:
                assign_per_img.append(item)

        if len(assign_per_img) == 0:
            probs_map = np.zeros(tuple([int(i / 2**level_save) for i in slide.level_dimensions[0]]))
        else:
            dataloader = make_dataloader(
                args, cnn, slide, level_ckpt, assign_per_img, flip='NONE', rotate='NONE')
            probs_map, time_network = get_probs_map(model, slide, level_save, level_ckpt, dataloader, prior=first_stage_map)
            time_total += time_network
        
        # save heatmap
        probs_map = (probs_map * 255).astype(np.uint8)
        shape_save = tuple([int(i / 2**level_save) for i in slide.level_dimensions[0]])
        probs_map = cv2.resize(probs_map, (shape_save[1], shape_save[0]), interpolation=cv2.INTER_CUBIC)
        np.save(os.path.join(args.probs_path, 'model_l{}'.format(level_ckpt), \
                                 'save_dyn_l{}'.format(level_save), file.split('.')[0] + '.npy'), probs_map)

        # visulize heatmap
        img_rgb = slide.read_region((0, 0), level_show, \
                            tuple([int(i/2**level_show) for i in slide.level_dimensions[0]])).convert('RGB')
        img_rgb = np.asarray(img_rgb).transpose((1,0,2))
        probs_map = cv2.resize(probs_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        probs_img_rgb = cv2.applyColorMap(probs_map, cv2.COLORMAP_JET)
        probs_img_rgb = cv2.cvtColor(probs_img_rgb, cv2.COLOR_BGR2RGB)
        heat_img = cv2.addWeighted(probs_img_rgb.transpose(1,0,2), 0.5, img_rgb.transpose(1,0,2), 0.5, 0)
        cv2.imwrite(os.path.join(args.probs_path, 'model_l{}'.format(level_ckpt), \
                                   'save_dyn_l{}'.format(level_save), file.split('.')[0] + '_heat.png'), heat_img)

    time_total_avg = time_total / len(dir)
    logging.info('AVG Total Run Time : {:.2f}'.format(time_total_avg))

def main():
    args = parser.parse_args([
        "./datasets/test/images",
        "./save_train/train_dyn_l1",
        "./camelyon16/configs/cnn_dyn_l1.json",
        './datasets/test/dens_map_sampling_l8/model_l1/save_l3',
        "./datasets/test/crop_split_l1/assign.json",
        './datasets/test/dens_map_assign_l6'])
    args.GPU = "2"
    run(args)


if __name__ == '__main__':
    main()