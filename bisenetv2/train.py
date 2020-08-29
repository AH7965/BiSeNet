#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from bisenetv2.bisenetv2 import BiSeNetV2
from bisenetv2.cityscapes_cv2 import get_data_loader
from bisenetv2.evaluatev2 import eval_model
from bisenetv2.ohem_ce_loss import OhemCELoss
from bisenetv2.lr_scheduler import WarmupPolyLrScheduler
from bisenetv2.meters import TimeMeter, AvgMeter
from bisenetv2.logger import setup_logger, print_log_msg

# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False

print('has_apex', has_apex)


## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--dataset', dest='dataset', type=str, default='CityScape',choices=['CityScape', 'FaceMask'])
    parse.add_argument('--sync-bn', dest='use_sync_bn', action='store_true',)
    parse.add_argument('--fp16', dest='use_fp16', action='store_true',)
    parse.add_argument('--port', dest='port', type=int, default=44554,)
    parse.add_argument('--respth', dest='respth', type=str, default='./res',)
    parse.add_argument('--datapth', dest='datapth', type=str, default='./data',)
    parse.add_argument('--bts', dest='batch_size', type=int, default=8,)
    return parse.parse_args()

args = parse_args()

lr_start = 5e-2
warmup_iters = 1000
max_iter = 150000  + warmup_iters
ims_per_gpu = args.batch_size
dataset = args.dataset
datapth = args.datapth

def set_model():
    if dataset == 'CityScape':
        net = BiSeNetV2(19)
    elif dataset == 'FaceMask':
        net = BiSeNetV2(2)
    if args.use_sync_bn: net = set_syncbn(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(4)]
    return net, criteria_pre, criteria_aux

def set_syncbn(net):
    if has_apex:
        net = parallel.convert_syncbn_model(net)
    else:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def set_optimizer(model):
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if param.dim() == 1:
            non_wd_params.append(param)
        elif param.dim() == 2 or param.dim() == 4:
            wd_params.append(param)
    params_list = [
        {'params': wd_params, },
        {'params': non_wd_params, 'weight_decay': 0},
    ]
    optim = torch.optim.SGD(
        params_list,
        lr=lr_start,
        momentum=0.9,
        weight_decay=5e-4,
    )
    return optim


def set_model_dist(net):
    if has_apex:
        net = parallel.DistributedDataParallel(net, delay_allreduce=True)
    else:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank)
    return net


def set_meters():
    time_meter = TimeMeter(max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i)) for i in range(4)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def ohlabel2segmap(label):
    n_classes=label.shape[0]
    step=255//((n_classes-1)**(1/3))
    coloers = [np.array([i//9*step, i%9//3*step, i%3*step]) for i in range(0, n_classes)]
    np_label = label.detach().cpu().numpy().argmax(axis=0)
    im_label = np.zeros((label.shape[1], label.shape[2], 3))
    for i in range(n_classes):
        im_label[np_label==i] = coloers[i]
    return im_label.astype(np.int)
    

def label2segmap(label, n_classes=2):
    step=255//((n_classes-1)**(1/3))
    coloers = [np.array([i//9*step, i%9//3*step, i%3*step]) for i in range(0, n_classes)]
    np_label = label.detach().cpu().numpy()
    im_label = np.zeros((label.shape[0], label.shape[1], 3))
    for i in range(n_classes):
        im_label[np_label==i] = coloers[i]
    return im_label.astype(np.int)


def unnorm(image):
    np_image = ((image.detach().cpu().numpy().transpose(1, 2, 0) * np.array((0.2112, 0.2148, 0.2115)) + np.array((0.3257, 0.3690, 0.3223))).clip(0, 1)*255).astype(np.int)[:, :, ::-1]
    # print(np_image.shape, np_image.min(), np_image.max())
    return np_image


def make_sample(input, gt, result):
    inputs = cv2.hconcat([unnorm(i) for i in input])
    gts = cv2.hconcat([label2segmap(g) for g in gt])
    results = cv2.hconcat([ohlabel2segmap(r) for r in result])
    # print(inputs.shape, gts.shape, results.shape)
    # print(inputs.dtype, gts.dtype, results.dtype)
    # print(inputs.min(), gts.min(), results.min())
    # print(inputs.max(), gts.max(), results.max())
    return cv2.vconcat([inputs, gts, results])

def save_sample(save_pth, input, gt, result, key=None):
    # print(input.shape, gt.shape, result.shape)
    sample = make_sample(input, gt, result)
    # print(sample.shape)
    cv2.imwrite(osp.join(save_pth, f'sample_{key}.png'), sample)




def train():
    logger = logging.getLogger()
    is_dist = dist.is_initialized()

    ## dataset
    dl = get_data_loader(datapth, ims_per_gpu, max_iter,
            mode='train', distributed=is_dist, dataset=dataset)

    ## model
    net, criteria_pre, criteria_aux = set_model()

    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex and args.use_fp16:
        net, optim = amp.initialize(net, optim, opt_level='O1')

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=max_iter, warmup_iter=warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loop
    for it, (im, lb) in enumerate(dl):
        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        logits, *logits_aux = net(im)
        loss_pre = criteria_pre(logits, lb)
        loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)
        if has_apex:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optim.step()
        torch.cuda.synchronize()
        lr_schdr.step()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        ## print training log message
        if (it) % 10000 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(
                it, max_iter, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)

            os.makedirs(osp.join(args.respth, 'sample'), exist_ok=True)
            save_sample(osp.join(args.respth, 'sample'), im, lb, logits, it)
            state = net.module.state_dict()
            torch.save(state, osp.join(args.respth, f'model_{it}.pth'))
            eval_model(net, 4)

    ## dump the final model and evaluate the result
    save_pth = osp.join(args.respth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0: torch.save(state, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    eval_model(net, 4)

    return


def main():
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    if not osp.exists(args.respth): os.makedirs(args.respth)
    setup_logger('BiSeNetV2-train', args.respth)
    train()


if __name__ == "__main__":
    main()
