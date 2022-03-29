# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: SiamRPN onekey script
# ------------------------------------------------------------------------------

import _init_paths
import os
import yaml
import argparse
from os.path import exists
from utils.utils import load_yaml, extract_logs

def parse_args():
    """
    args for onekey.
    """
    parser = argparse.ArgumentParser(description='Train SiamISO with onekey')
    # for train
    parser.add_argument('--cfg', type=str, default='./experiments/eval/SiamISO.yaml', help='yaml configure file name')

    # for

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # train - test - tune information
    f = open(args.cfg, 'r').read()
    info = yaml.load(f,Loader=yaml.FullLoader)
    start_epoch = info['START_EPOCH']
    end_epoch = info['END_EPOCH']
    thread = info['THREADS']
    os.system('python ./siamese_tracking/eva_epochs.py --start_epoch {0} --end_epoch {1} --threads {2}'.format(start_epoch, end_epoch, thread))


if __name__ == '__main__':
    main()