# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# multi-gpu test for epochs
# ------------------------------------------------------------------------------

import sys
import os
import time
import argparse
from mpi4py import MPI


parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
parser.add_argument('--arch', dest='arch', default='SiamFCIncep22',
                    help='architecture of model')
parser.add_argument('--start_epoch', default=30, type=int, required=True, help='test end epoch')
parser.add_argument('--end_epoch', default=50, type=int, required=True,
                    help='test end epoch')
parser.add_argument('--gpu_nums', default=4, type=int, required=True, help='test start epoch')
parser.add_argument('--anchor_nums', default=5, type=int, help='anchor numbers')
parser.add_argument('--threads', default=16, type=int, required=True)
parser.add_argument('--dataset', default='OTB2013', type=str, help='benchmark to test')
parser.add_argument('--result_path', default='./result_iso', type=str, help='multi-gpu epoch test flag')
args = parser.parse_args()

# init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() # 线程编号
GPU_ID = rank % args.gpu_nums # 线程在哪个GPU上跑
node_name = MPI.Get_processor_name()  # get the name of the node
# node_name = MPI.name
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
time.sleep(rank * 5)

flag = 0
# run test scripts -- two epoch for each thread，每个MPI运行两个epoch
for i in range(1):
    arch = args.arch
    dataset = args.dataset
    try:

        # 测试一个epoch多线程
        epoch_ID += 0
        flag += 1
        # epoch_ID += args.threads   # for 16 queue
    except:
        # 这一句相当于每个MDI thread的epoch_ID的初始化
        epoch_ID = rank % (args.end_epoch - args.start_epoch + 1) + args.start_epoch
        flag += 1

    if epoch_ID > args.end_epoch:
        continue

    if flag > 11:
        continue

    # 对应模型地址
    resume = 'snapshot_iso/checkpoint_e{}.pth'.format(epoch_ID)
    # print('==> test {}th epoch'.format(epoch_ID))


    # 把模型放进去进行测试
    if 'SiamFC' in arch:
        os.system('python ./siamese_tracking/test_siamfc.py --arch {0} --resume {1} --dataset {2} --epoch_test True'.format(arch, resume, dataset))
    
    
    if 'SiamRPN' in arch:
        os.system('python ./siamese_tracking/test_siamrpn.py --arch {0} --resume {1} --dataset {2}  --epoch_test True'.format(arch, resume, dataset))

    if 'ISO' in arch:
        os.system('python ./siamese_tracking/test_siamiso2.py --arch {0} --resume {1} --dataset {2} --result_path {3} --epoch_test True'.format(arch, resume, dataset,args.result_path))
