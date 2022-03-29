import _init_paths
import models.modelIso as models
from models.modelIso import TestNet
from utils.utils import load_pretrain
import os
import glob
import random
import cv2
import numpy as np
import time

import torch
import torchvision.transforms as transforms

import argparse
from os.path import join, realpath, dirname, exists

import pandas as pd
from core.config import config
import os

def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamISO Tracking Test')
    parser.add_argument('--arch', dest='arch', default='SiamFCIncep22', help='backbone architecture')
    parser.add_argument('--resume', required=True, type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2017', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    # parser.add_argument('--result_path', default=False, type=bool, help='result path')
    args = parser.parse_args()

    return args

def load_gt(path):
    gt_path = os.path.join(path, 'sot2')
    target_gt = {}
    for root, _, files in os.walk(gt_path):
        for f in files:
            target_gt_path = os.path.join(root,f)
            # print(target_gt_path)
            target_index = int(f.split(".")[0])
            gt = pd.read_table(target_gt_path,sep=',',header=None)
            gt.columns = ['x','y','w','h','frame']
            
            gt = gt.set_index(['frame'])
            gt = gt.sort_index()
            
            target_gt[target_index] = gt.iloc[0,:]
            target_gt[target_index].name = str(target_gt[target_index].name).zfill(6) + ".jpg"

    return target_gt


def load_dataset(path):
    """
    support OTB and VOT now
    TODO: add other datasets
    {video:{target:dataframe}}
    dataframe.index = target.index
    dataframe.columns = target_first_frame_gt
    """
    dataset = {}
    val_video_path = [] # 存放每个video文件夹路径
    val_video = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            val_video_path.append(os.path.join(root, d))
            val_video.append(d)
        break
    
    for v_path in val_video_path:
        target_gt  = load_gt(v_path)
        dataset[v_path] = target_gt

    return dataset


def get_region(offset, img_size, img, x, y):
    # print(x,y,img.shape)
    if x <= offset <= y and y + offset < img_size:
        # print(x,y,offset,img_size)
        img_region = img[y - offset:y + offset + 1, 0:2 * x + 1]
        # print(1)
    elif x <= offset <= y and y + offset >= img_size:
        img_region = img[2 * y - img_size:img_size, 0:2 * x + 1]
        # print(2)
    elif y <= offset <= x and x + offset < img_size:
        img_region = img[0:2 * y + 1, x - offset:x + offset + 1]
        # print(3)
    elif y <= offset <= x and x + offset >= img_size:
        img_region = img[0:2 * y + 1, 2 * x - img_size:img_size]
        # print(4)
    elif y + offset >= img_size and x + offset >= img_size:
        img_region = img[2 * y - img_size:img_size, 2 * x - img_size:img_size]
        # print(5)
    elif x < offset and y < offset:
        img_region = img[0:2 * y + 1, 0:2 * x + 1]
        # print(6)
    else:
        img_region = img[y - offset:y + offset + 1, x - offset:x + offset + 1]
        # print(7)

    return img_region


def pad_image(image, out_size):
    y = image.shape[0]
    x = image.shape[1]

    n_pad_x = out_size - x
    n_pad_y = out_size - y
    n_pad_x0 = int(n_pad_x / 2)
    n_pad_x1 = n_pad_x - n_pad_x0
    n_pad_y0 = int(n_pad_y / 2)
    n_pad_y1 = n_pad_y - n_pad_y0

    # print(x,y,out_size,n_pad_y0,n_pad_y1,n_pad_x0,n_pad_x1)
    # print(image.shape)
    padded = np.pad(image, ((n_pad_y0, n_pad_y1), (n_pad_x0, n_pad_x1)),
                    'mean')
        

    return padded


if __name__ == '__main__':
    template_size = 13  # 模板大小
    template_offset = int(template_size / 2)
    search_size = 65  # 搜索区域大小
    stride = 1
    search_offset = int(search_size / 2)
    center = search_offset + 1
    img_size = 1024  # 图像大小
    ignore_size = 16
    update_fre = 100
    displace_th = 32


    # 加载模型
    backbone = TestNet()
    args = parse_args()
    net = models.__dict__[
        config.SIAMISO.TRAIN.MODEL](embedding_net=backbone) 
    net = load_pretrain(net, args.resume)
    # net = models.__dict__[
    #     "SiameseISO"](embedding_net=backbone) 
    # net = load_pretrain(net, "./snapshot_iso/checkpoint_e6.pth")
    net.eval() # 不启用batch normalization和dropout
    net = net.cuda()

    # 加载数据
    dataset = load_dataset("/mnt3/lichenhao/VISO/validation data/")
    video_keys = list(dataset.keys()).copy()
    # print(video_keys)

    transform = transforms.ToTensor()

    # data_path = './infrared_small_object_data/val'

    # 余弦窗
    hanning = np.hanning((search_size-template_size)/stride+1)
    hanning = np.outer(hanning, hanning)

    result_path = './result_iso'
    if not exists(result_path):
        os.makedirs(result_path)
    
    train_epoch_name = args.resume.split("/")[-1]
    # train_epoch_name = "6"
    result_path = os.path.join(result_path,train_epoch_name)
    if not exists(result_path):
        os.makedirs(result_path)


    for v_path in video_keys:
        video_path = os.path.join("/mnt3/lichenhao/VISO/validation data",str(v_path))
        videos_path = os.path.join(video_path,'img1')
        video_target_frame = dataset[v_path]
        anno = []
        track = 0
        score = 0

        v_name = v_path.split("/")[-1]
        result_video_path = os.path.join(result_path,v_name)
        if not exists(result_video_path):
            os.makedirs(result_video_path)

        for tar, gt in video_target_frame.items():

            track_flag = 0
            result_tar_path = result_video_path+'/' + str(tar) + '.txt'
            with open(result_tar_path, "w") as fin:
            # img为图片名
                imgs_path = sorted(os.listdir(videos_path))
                for img in imgs_path:
                    img_path = os.path.join(videos_path,img)
                    # result_txt_path = os.path.join()
                    if track_flag != 0:
                        time_start = time.time()
                        # print(img_path)
                        o_search = np.array(cv2.imread(img_path, 0), dtype=np.uint8)

                        # 在中心裁剪区域
                        img_size = o_search.shape[0]
                        search = get_region(search_offset, img_size, o_search, x_t, y_t)
                        # padding到搜索大小
                        search = pad_image(search, search_size)
                        search = cv2.cvtColor(search,cv2.COLOR_GRAY2RGB)
                        

                        search = transform(search)
                        search = search.cuda()
                        search = torch.unsqueeze(search, 0)

                        # 输出结果
                        output = net(template,search)

                        # 得到score_map
                        output = torch.squeeze(output, 0)
                        output = torch.squeeze(output, 0)

                        # 归一化分数
                        prediction = output.cpu().detach().numpy()
                        prediction -= np.min(prediction)
                        prediction /= np.max(prediction)

                        # 余弦窗
                        prediction = hanning * prediction
                        # 裁剪忽略的边缘部分
                        prediction = prediction[ignore_size:-ignore_size,
                                                ignore_size:-ignore_size]
                        position = np.unravel_index(np.argmax(prediction),
                                                    prediction.shape)

                        # 计算偏移
                        displace_x = position[1] - center + ignore_size + 1
                        displace_y = position[0] - center + ignore_size + 1

    
                        x_t += displace_x
                        y_t += displace_y

                        x_t = max(0,x_t)
                        x_t = min(img_size-1,x_t)
                        y_t = max(0,y_t)
                        y_t = min(img_size-1,y_t)
                        
                        fin.write(','.join([str(x_t),str(y_t),str(w),str(h),img]))
                        fin.write('\n')


                        # 计算时间
                        time_end = time.time()
                        print("fps：%f",1 / (time_end - time_start))

                    if str(gt.name) == img: 
                    # 图片路径
                        # print("tar",tar)
                        # print("video",v_path)
                        template = np.array(cv2.imread(img_path, 1), dtype=np.uint8)

                        # 从左上角坐标转到中心
                        x_t = int(gt.iloc[0] + gt.iloc[2] / 2) 
                        y_t = int(gt.iloc[1] + gt.iloc[3] / 2)
                        w,h = gt.iloc[2],gt.iloc[3]

                        template_offset = int(template_size / 2)
                        search_offset = int(search_size / 2)

                        template = np.array(cv2.imread(img_path, 0), dtype=np.uint8)
                        # 裁剪模板
                        template_top = max(y_t - template_offset, 0)
                        template_bottom = min(y_t + template_offset + 1, template.shape[0]-1)
                        template_left = max(x_t - template_offset, 0)
                        template_right = min(x_t + template_offset + 1, template.shape[1]-1)
                        template = template[template_top:template_bottom,
                                            template_left:template_right]
                        # print(x_t,y_t,template_offset)
                        # print(template.shape)
                        template = pad_image(template, template_size)
                        template = cv2.cvtColor(template,cv2.COLOR_GRAY2RGB)
                        template = transform(template)
                        template = template.cuda()
                        template = torch.unsqueeze(template, 0)
                        
                        track_flag = 1



    print("44444444444444444444444444444444444")
