# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and  Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Details: SIAMISO training script
# ------------------------------------------------------------------------------
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
from sklearn import utils
from torch import embedding
import _init_paths
import os
import shutil
import time
import math
import pprint
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

import models.modelIso as models
from models.modelIso import TestNet

from utils.utils import create_logger, print_speed, load_pretrain, restore_from, save_model
from dataset.siamiso import SiamISODataset, SiamISODatasetWithNeg
from core.config import config, update_config
from core.function import siamiso_train

eps = 1e-5


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train SIAMISO')
    # general
    parser.add_argument('--cfg',
                        required=True,
                        type=str,
                        default='experiments/train/SIAMISO.yaml',
                        help='yaml configure file name')

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers',
                        type=int,
                        help='num of dataloader workers')

    args = parser.parse_args()

    return args


def reset_config(config, args):
    """
    set gpus and workers
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def check_trainable(model, logger):
    """
    print trainable params info
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'

    return trainable_params


def get_optimizer(cfg, trainable_params):
    """
    get optimizer
    """

    # optimizer = torch.optim.SGD(trainable_params,
    #                             cfg.SIAMISO.TRAIN.LR,
    #                             momentum=cfg.SIAMISO.TRAIN.MOMENTUM,
    #                             weight_decay=cfg.SIAMISO.TRAIN.WEIGHT_DECAY)

    optimizer = torch.optim.Adam(trainable_params,
                            lr=cfg.SIAMISO.TRAIN.LR1,
                            betas=(cfg.SIAMISO.TRAIN.BETAS1,cfg.SIAMISO.TRAIN.BETAS2),
                            eps=1e-08
                            )

    return optimizer


def lr_decay(cfg, optimizer):
    if cfg.SIAMISO.TRAIN.LR_POLICY == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.8685)
    elif cfg.SIAMISO.TRAIN.LR_POLICY == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.SIAMISO.TRAIN.LR_POLICY == 'Reduce':
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
    elif cfg.SIAMISO.TRAIN.LR_POLICY == 'log':
        scheduler = np.logspace(math.log10(cfg.SIAMISO.TRAIN.LR),
                                math.log10(cfg.SIAMISO.TRAIN.LR_END),
                                cfg.SIAMISO.TRAIN.END_EPOCH)

    else:
        raise ValueError('unsupported learing rate scheduler')

    return scheduler


def pretrain_zoo():
    GDriveIDs = dict()
    GDriveIDs['SiamFCRes22'] = "1kgYJdydU7Wm6oj9-tGA5EFc6Io2V7rPT"
    GDriveIDs['SiamFCIncep22'] = "1FxbQOSsG51Wau6-MUzsteoald3Y14xJ4"
    GDriveIDs['SiamFCNext22'] = "1sURid92u4hEHR4Ev0wrQPAw8GZtLmB5n"
    return GDriveIDs


def adjust_lr(optimizer,cfg):
    for para in optimizer.param_groups:
        para['lr'] = cfg.SIAMISO.TRAIN.LR2
    return optimizer

def main():
    # [*] args, loggers and tensorboard
    args = parse_args()
    reset_config(config, args)

    logger, _, tb_log_dir = create_logger(config, 'SIAMISO', 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    # auto-download train model from GoogleDrive
    if not os.path.exists('./pretrain'):
        os.makedirs('./pretrain')
    try:
        DRIVEID = pretrain_zoo()

        # 下载预训练模型
        if not os.path.exists('./pretrain/{}'.format(
                config.SIAMISO.TRAIN.PRETRAIN)):
            os.system(
                'wget --no-check-certificate \'https://drive.google.com/uc?export=download&id={0}\' -O ./pretrain/{1}'
                .format(DRIVEID[config.SIAMISO.TRAIN.MODEL],
                        config.SIAMISO.TRAIN.PRETRAIN))
    except:
        print(
            'auto-download pretrained model fail, please download it and put it in pretrain directory'
        )

    # [*] gpus parallel and model prepare
    # prepare
    backbone = TestNet()
    model = models.__dict__[
        config.SIAMISO.TRAIN.MODEL](embedding_net=backbone)  # build model，将模型信息变成字典的形式
    # model = load_pretrain(model, './pretrain/{}'.format(
    #     config.SIAMISO.TRAIN.PRETRAIN))  # load pretrain
    trainable_params = check_trainable(
        model, logger)  # print trainable params info，将可训练的参数保存下来
    optimizer = get_optimizer(config, trainable_params)  # optimizer，定义优化器
    lr_scheduler = lr_decay(config,
                            optimizer)  # learning rate decay scheduler，下降速率

    if config.SIAMISO.TRAIN.RESUME and config.SIAMISO.TRAIN.START_EPOCH != 0:  # resume
        model.features.unfix((config.SIAMISO.TRAIN.START_EPOCH - 1) /
                             config.SIAMISO.TRAIN.END_EPOCH)
        model, optimizer, args.start_epoch, arch = restore_from(
            model, optimizer, config.SIAMISO.TRAIN.RESUME)

    # parallel，并行数量
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))
    gpu_ids = [i for i in range(gpu_num)]
    # multiGPU
    model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    logger.info('model prepare done')

    # [*] train
    half_epoch = (config.SIAMISO.TRAIN.START_EPOCH+config.SIAMISO.TRAIN.END_EPOCH) // 2
    tensor_writer = SummaryWriter('./tensorborad')
    for epoch in range(config.SIAMISO.TRAIN.START_EPOCH,
                       config.SIAMISO.TRAIN.END_EPOCH):
        if epoch == half_epoch:
            optimizer = adjust_lr(optimizer,config)

        # build dataloader, benefit to tracking
        # train_set = SiamISODataset(config)
        train_set = SiamISODatasetWithNeg(config)
        train_loader = DataLoader(train_set,
                                  batch_size=config.SIAMISO.TRAIN.BATCH*gpu_num,
                                  num_workers=config.WORKERS,
                                  pin_memory=True,
                                  sampler=None)

        # train_loader = DataLoader(train_set,
        #                     batch_size=config.SIAMISO.TRAIN.BATCH *
        #                     gpu_num,
        #                     num_workers=config.WORKERS,
        #                     pin_memory=True,
        #                     sampler=None)

        # if config.SIAMISO.TRAIN.LR_POLICY == 'log':
        #     curLR = lr_scheduler[epoch]
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = curLR
        # else:
        #     lr_scheduler.step()

        if epoch < half_epoch:
            curLR = config.SIAMISO.TRAIN.LR1
        else:
            curLR = config.SIAMISO.TRAIN.LR2

        model, writer_dict = siamiso_train(train_loader, model, optimizer,
                                          epoch + 1, curLR, config,
                                          writer_dict, logger,tensor_writer)

        # save model
        save_model(model,
                   epoch,
                   optimizer,
                   config.SIAMISO.TRAIN.MODEL,
                   config,
                   isbest=False)
    
    tensor_writer.close()
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()