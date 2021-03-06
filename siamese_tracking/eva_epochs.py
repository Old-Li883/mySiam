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
parser.add_argument('--start_epoch', default=30, type=int, required=True, help='test end epoch')
parser.add_argument('--end_epoch', default=50, type=int, required=True,
                    help='test end epoch')
parser.add_argument('--threads', default=16, type=int, required=True)
args = parser.parse_args()

# init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() # 线程编号
# GPU_ID = rank % 4 # 线程在哪个GPU上跑
node_name = MPI.Get_processor_name()  # get the name of the node
# os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

# print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
time.sleep(rank * 5)


# run test scripts -- two epoch for each thread，每个MPI运行两个epoch
for i in range(args.end_epoch - args.start_epoch + 1):
    try:
        # epoch_ID += args.threads   # for 16 queue
        epoch_ID += 1
    except:
        # 这一句相当于每个MDI thread的epoch_ID的初始化
        epoch_ID = rank % (args.end_epoch - args.start_epoch + 1) + args.start_epoch

    if epoch_ID > args.end_epoch:
        continue

    print('==> evl {}th epoch'.format(epoch_ID))


    # 把模型放进去进行测试

    os.system('python ./siamese_tracking/eval_iso.py --epoch {0}'.format(epoch_ID))

