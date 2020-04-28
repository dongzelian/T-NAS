import numpy as np
import scipy.stats
import argparse
import os
import logging
import glob
import sys
import time
import random


from MiniImagenet import MiniImagenet
from meta_auto_maml_train import Meta
import utils.utils as utils
from utils.saver import Saver
from utils.summaries import TensorboardSummary

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info

import pdb

parser = argparse.ArgumentParser("mini-imagenet")
parser.add_argument('--dataset', type=str, default='mini-imagenet', help='dataset')
parser.add_argument('--checkname', type=str, default='auto-maml-train', help='checkname')
parser.add_argument('--run', type=str, default='run_auto_maml', help='run_path')
parser.add_argument('--data_path', type=str, default='', help='path to data')
parser.add_argument('--seed', type=int, default=222, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epoch', type=int, help='epoch number', default=10)
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
parser.add_argument('--n_way', type=int, help='n way', default=5)
parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=600, help='test batch size')
parser.add_argument('--meta_batch_size', type=int, help='meta batch size, namely task num', default=4)
parser.add_argument('--meta_test_batch_size', type=int, help='meta test batch size', default=1)
parser.add_argument('--report_freq', type=float, default=30, help='report frequency')
parser.add_argument('--test_freq', type=float, default=500, help='test frequency')
parser.add_argument('--img_size', type=int, help='img_size', default=84)
parser.add_argument('--imgc', type=int, help='imgc', default=3)
parser.add_argument('--meta_lr_w', type=float, help='meta-level outer learning rate (w)', default=1e-3)
parser.add_argument('--update_lr_w', type=float, help='task-level inner update learning rate (w)', default=0.01)
parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--arch', type=str, default='AUTO_MAML_2', help='which architecture to use')
parser.add_argument('--pretrained_model', type=str, default='', help='path to pretrained model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


args = parser.parse_args()

best_pred = 0

def main():
    saver = Saver(args)
    # set log
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p',
                        filename=os.path.join(saver.experiment_dir, 'log.txt'), filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled=True

    # set saver
    saver.create_exp_dir(scripts_to_save=glob.glob('*.py') + glob.glob('*.sh') + glob.glob('*.yml'))
    saver.save_experiment_config()
    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()


    logging.info(args)

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    ''' Compute FLOPs and Params '''
    maml = Meta(args, criterion)
    flops, params = get_model_complexity_info(maml.model, (3, 84, 84), as_strings=False, print_per_layer_stat=True, verbose=True)
    logging.info('FLOPs: {} MMac Params: {}'.format(flops / 10 ** 6, params))

    maml = Meta(args, criterion).to(device)
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    #logging.info(maml)
    logging.info('Total trainable tensors: {}'.format(num))

    # batch_size here means total episode number
    mini = MiniImagenet(args.data_path, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batch_size=args.batch_size, resize=args.img_size)
    mini_test = MiniImagenet(args.data_path, mode='val', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batch_size=args.test_batch_size, resize=args.img_size)
    train_loader = DataLoader(mini, args.meta_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(mini_test, args.meta_test_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # load pretrained model and inference
    if args.pretrained_model:
        checkpoint = torch.load(args.pretrained_model)
        if isinstance(maml.model, torch.nn.DataParallel):
            maml.module.load_state_dict(checkpoint['state_dict'])
        else:
            maml.load_state_dict(checkpoint['state_dict'])

        if args.evaluate:
            test_accs = meta_test(test_loader, maml, device, checkpoint['epoch'])
            logging.info('[Epoch: {}]\t Test acc: {}'.format(checkpoint['epoch'], test_accs))
            return


    # Start training
    for epoch in range(args.epoch):
        # fetch batch_size num of episode each time
        logging.info('--------- Epoch: {} ----------'.format(epoch))

        train_accs = meta_train(train_loader, maml, device, epoch, writer, test_loader, saver)
        logging.info('[Epoch: {}]\t Train acc: {}'.format(epoch, train_accs))


def meta_train(train_loader, maml, device, epoch, writer, test_loader, saver):
    global best_pred
    accs_all_train = []
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    update_w_time = utils.AverageMeter()
    end = time.time()
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_loader):
        data_time.update(time.time() - end)
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        accs, update_w_time = maml(x_spt, y_spt, x_qry, y_qry, update_w_time)
        accs_all_train.append(accs)
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('train/acc_iter', accs[-1], step + len(train_loader) * epoch)
        if step % args.report_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'W {update_w_time.val:.3f} ({update_w_time.avg:.3f})\t'
                         'training acc: {accs}'.format(
                    epoch, step, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    update_w_time=update_w_time, accs=accs))

        if step % args.test_freq == 0:
            test_accs = meta_test(test_loader, maml, device, epoch)
            writer.add_scalar('val/acc', test_accs[-1], step // args.test_freq + (len(train_loader) // args.test_freq + 1) * epoch)
            logging.info('[Epoch: {}]\t Test acc: {}'.format(epoch, test_accs))

            # Save the best meta model.
            new_pred = test_accs[-1]
            if new_pred > best_pred:
                is_best = True
                best_pred = new_pred
            else:
                is_best = False
            saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': maml.module.state_dict() if isinstance(maml, nn.DataParallel) else maml.state_dict(),
                'best_pred': best_pred,
            }, is_best)

    accs = np.array(accs_all_train).mean(axis=0).astype(np.float16)

    return accs


def meta_test(test_loader, maml, device, epoch):
    accs_all_test = []
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    update_w_time = utils.AverageMeter()
    end = time.time()
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(test_loader):
        data_time.update(time.time() - end)
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        accs, update_w_time = maml.finetunning(x_spt, y_spt, x_qry, y_qry, update_w_time)
        accs_all_test.append(accs)
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'W {update_w_time.val:.3f} ({update_w_time.avg:.3f})\t'
                         'test acc: {accs}'.format(
                    epoch, step, len(test_loader),
                    batch_time=batch_time, data_time=data_time,
                    update_w_time=update_w_time,accs=accs))

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)  # accs.shape=11

    return accs


if __name__ == '__main__':
    main()
