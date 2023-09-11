import sys
import os
import argparse
import logging
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
from torchvision import models
from torch import nn

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camelyon16.data.image_producer import ImageDataset

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=0, type=int, help='number of'
                    ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0', type=str, help='comma'
                    ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                    ' and GPU_1, default 0.')
parser.add_argument('--init_mode', default='finetune', type=str, help='comma'
                    ' how to initial the pretrained weight')

def chose_model(cnn):
    if cnn['model'] == 'segmentation':
        model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
    else:
        raise Exception("I have not add any models. ")
    return model


def train_epoch(summary, summary_writer, cnn, model, loss_fn, optimizer, dataloader):
    
    model.train()
    dataloader.dataset.data_split('train')
    dataloader.dataset.data_shuffle()
    steps = len(dataloader)
    dataiter_train = iter(dataloader)

    time_now = time.time()
    for step in range(steps):
        data_train, target_train = next(dataiter_train)
        data_train = Variable(data_train.float().cuda(non_blocking=True))
        target_train = Variable(target_train.float().cuda(non_blocking=True))

        output = model(data_train)
        output = output['out'][:, 0].flatten() # noqa
        target_train = target_train.flatten()
        loss = loss_fn(output, target_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = output.sigmoid()
        predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)

        acc_data = (predicts == target_train).type(
            torch.cuda.FloatTensor).sum().data * 1.0 / len(target_train)
        loss_data = loss.data

        time_spent = time.time() - time_now
        logging.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                summary['step'] + 1, loss_data, acc_data, time_spent))

        summary['step'] += 1

        if summary['step'] % cnn['log_every'] == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data, summary['step'])

    summary['epoch'] += 1

    return summary


def valid_epoch(summary, model, loss_fn, dataloader):
    
    model.eval()
    dataloader.dataset.data_split('valid')
    dataloader.dataset.data_shuffle()
    steps = len(dataloader)
    dataiter_valid = iter(dataloader)

    loss_sum = 0
    acc_sum = 0
    with torch.no_grad():
        for step in range(steps):
            data_valid, target_valid = next(dataiter_valid)
            data_valid = Variable(data_valid.float().cuda(non_blocking=True))
            target_valid = Variable(target_valid.float().cuda(non_blocking=True))

            output = model(data_valid)
            output = output['out'][:, 0].flatten() # noqa
            target_valid = target_valid.flatten()
            loss = loss_fn(output, target_valid)

            probs = output.sigmoid()
            predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
            acc_data = (predicts == target_valid).type(
                torch.cuda.FloatTensor).sum().data * 1.0 / len(target_valid)
            loss_data = loss.data

            loss_sum += loss_data
            acc_sum += acc_data

        summary['loss'] = loss_sum / steps
        summary['acc'] = acc_sum / steps

    return summary


def run(args):
    with open(args.cnn_path, 'r') as f:
        cnn = json.load(f)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(os.path.join(args.save_path, 'cnn.json'), 'w') as f:
        json.dump(cnn, f, indent=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))
    batch_size_train = cnn['batch_size'] * num_GPU
    batch_size_valid = cnn['batch_size'] * num_GPU
    num_workers = args.num_workers * num_GPU

    model = chose_model(cnn)
    # fc_features = model.fc.in_features
    # model.fc = nn.Linear(fc_features, 1)

    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    if args.init_mode == 'resume':
        checkpoint = torch.load(os.path.join(args.save_path, 'best.ckpt'))
        summary_train = {'epoch': checkpoint['epoch'], 'step': checkpoint['step']}
        model.load_state_dict(checkpoint['state_dict'])
    elif args.init_mode == 'finetune':
        checkpoint = torch.load('./save_train/train_base_l1/best.ckpt')
        model.load_state_dict(checkpoint['state_dict'])


    model = DataParallel(model, device_ids=None)
    model = model.cuda()

    loss_fn = BCEWithLogitsLoss().cuda()
    optimizer = SGD(model.parameters(), lr=cnn['lr'], momentum=cnn['momentum'])

    dataset = ImageDataset(cnn['data_path_train'], cnn['normalize'])

    dataloader= DataLoader(dataset, batch_size=batch_size_train, num_workers=num_workers)

    summary_writer = SummaryWriter(args.save_path)
    loss_valid_best = float('inf')
    for epoch in range(summary_train['epoch'], cnn['epoch']):
        summary_train = train_epoch(summary_train, summary_writer, cnn, 
                                    model, loss_fn, optimizer, dataloader)
        if epoch >= 0 and epoch % 1 ==0:
            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.module.state_dict()},
                        os.path.join(args.save_path, 'train_e{}.ckpt'.format(epoch)))

        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, model, loss_fn, dataloader)
        time_spent = time.time() - time_now

        logging.info('{}, Epoch: {}, step: {}, Validation Loss: {:.5f}, '
                     'Validation ACC: {:.3f}, Run Time: {:.2f}'
                     .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                             summary_train['step'], summary_valid['loss'],
                             summary_valid['acc'], time_spent))

        summary_writer.add_scalar('valid/loss',
                                  summary_valid['loss'], summary_train['step'])
        summary_writer.add_scalar('valid/acc',
                                  summary_valid['acc'], summary_train['step'])

        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.module.state_dict()},
                       os.path.join(args.save_path, 'best.ckpt'))

    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args([
        "./camelyon16/configs/cnn_dyn_l1.json",
        "./save_train/train_dyn_nobg_l1"])
    args.device_ids = '0'
    args.init_mode = 'resume'
    run(args)
    
    # args = parser.parse_args([
    #     "./camelyon16/configs/cnn_fix_l0.json",
    #     "./save_train/train_fix_l0"])
    # args.device_ids = '3'
    # args.init_mode = ''
    # run(args)

if __name__ == '__main__':
    main()
