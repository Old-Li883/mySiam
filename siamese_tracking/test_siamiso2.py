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
from torch.autograd import Variable

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
    parser.add_argument('--result_path', default='./result_iso', type=str, help='multi-gpu epoch test flag')
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

            num = int(gt.iloc[-1,:].name) - int(gt.iloc[0,:].name) + 1
            
            target_gt[target_index] = [gt.iloc[0,:],num]
            target_gt[target_index][0].name = str(target_gt[target_index][0].name).zfill(6) + ".jpg"

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
        dataset[v_path] = target_gt # 目标gt总数

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


def to_torch(ndarray):
    return torch.from_numpy(ndarray)
    

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img



def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    """
    SiamFC type cropping
    在pos附近取model_sz大小的窗口，可以是放大填充，也可以是裁剪
    im:模板图原始大小
    model_sz:孪生网络中模板大小
    original_sz:需要裁剪或者缩放的模板大小
    """
    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2

    # 计算将im模板图缩放到coriginal_sz所需要的padding数
    # 将原始坐标移动original_sz/2（模板图或者搜索图的一半，目标默认在中心）的距离，计算得出在旁边padding的大小
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    # 重模板图缩放后大小
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    # 填充通道平均值
    if any([top_pad, bottom_pad, left_pad, right_pad]): # padding都大于0则是说明要填充
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im # 将原图放进去
        # 填充均值
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else: # 否则是裁剪图片
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    # 改变大小到model_size
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    return im_to_torch(im_patch.copy()) if out_mode in 'torch' else im_patch




if __name__ == '__main__':
    template_size = 25  # 模板大小
    template_offset = int(template_size / 2)
    search_size = 65  # 搜索区域大小
    stride = 1
    search_offset = int(search_size / 2)
    center = search_offset + 1
    img_size = 1024  # 图像大小
    ignore_size = 16
    update_fre = 100
    displace_th = 32
    score_size = (search_size - template_size) / stride + 1
    w_influence = 0.35


    # 加载模型
    backbone = TestNet()

    # run
    args = parse_args()
    net = models.__dict__[
        config.SIAMISO.TRAIN.MODEL](embedding_net=backbone) 
    net = load_pretrain(net, args.resume)

    # test
    # net = models.__dict__[
    #     "SiameseISO"](embedding_net=backbone) 
    # net = load_pretrain(net, "./snapshot_iso/checkpoint_e47.pth")



    net.eval() # 不启用batch normalization和dropout
    net = net.cuda()

    # 加载数据
    dataset = load_dataset("/mnt3/lichenhao/VISO/validation data/")
    video_keys = list(dataset.keys()).copy()
    # print(video_keys)

    transform = transforms.ToTensor()

    # data_path = './infrared_small_object_data/val'

    # 余弦窗
    window = np.outer(np.hanning(int(score_size)),
                            np.hanning(int(score_size)))
    window /= window.sum()

    # result_path = './result_iso' # test
    result_path = args.result_path # run
    if not exists(result_path):
        os.makedirs(result_path)
    
    train_epoch_name = args.resume.split("/")[-1] # run
    # train_epoch_name = "47" # test

    result_path = os.path.join(str(result_path),str(train_epoch_name))
    if not exists(result_path):
        os.makedirs(result_path)

    # print(result_path,flush=True)
    #加上时间，test
    # nowtime = time.time()
    # result_path = os.path.join(result_path, str(nowtime))
    # if not exists(result_path):
    #     os.makedirs(result_path)

    scale_z = 1
    s_z = 0
    for v_path in video_keys:
        video_path = os.path.join("/mnt3/lichenhao/VISO/validation data",str(v_path))
        videos_path = os.path.join(video_path,'img1')
        video_target_frame = dataset[v_path]
        anno = []
        track = 0
        score = 0

        v_name = v_path.split("/")[-1] # video_name
        # result_video_path = os.path.join(result_path,v_name)
        # if not exists(result_video_path):
        #     os.makedirs(result_video_path)
            # print("test {0} video".format(v_name))

        for tar, gt in video_target_frame.items():

            track_flag = 0
            result_tar_path = result_path+'/' +str(v_name) + '_'+ str(tar)

            if not exists(result_tar_path):
                os.makedirs(result_tar_path)
                img_path = os.path.join(result_tar_path,'img')
                if not exists(img_path):
                    os.makedirs(img_path)

                result_tar_path_txt = result_tar_path + '/groundtruth_rect.txt'
                with open(result_tar_path_txt, "w") as fin:
                # img为图片名
                    imgs_path = sorted(os.listdir(videos_path))
                    count_flag = 0
                    for img in imgs_path:

                        # 计数
                        # if str(tar) == '24':
                        #     print('gt',gt[1])
                        #     print(count_flag)

                        
                        if count_flag >= gt[1]:
                            break

                        img_path = os.path.join(videos_path,img)
                        # result_txt_path = os.path.join()
                        if track_flag != 0:
                            time_start = time.time()
                            # print(img_path)
                            o_search = np.array(cv2.imread(img_path, 1), dtype=np.uint8)

                            avg_chans = np.mean(o_search, axis=(0, 1)) # 3通道平均值

                            d_search = (search_size - template_size) / 2
                            pad = d_search / scale_z
                            s_x = s_z + 2 * pad
                            
                            x_crop = get_subwindow_tracking(o_search, [x_t, y_t], search_size, s_x, avg_chans)
                            search = Variable(x_crop.unsqueeze(0)).cuda()                     


                            # 输出结果
                            output = net(template,search)

                            # 得到score_map
                            output = torch.squeeze(output, 0)
                            output = torch.squeeze(output, 0)


                            # 归一化分数
                            prediction = output.cpu().detach().numpy()
                            prediction -= np.min(prediction)
                            prediction /= prediction.sum()

                            # 余弦窗
                            prediction = (1 - w_influence) * prediction + w_influence * window
                            # 裁剪忽略的边缘部分
                            position = np.unravel_index(np.argmax(prediction),
                                                        prediction.shape)

                            # 计算偏移
                            displace_x = (position[1] - prediction.shape[1] // 2) * s_x / search_size
                            displace_y = (position[0] - prediction.shape[0] // 2) * s_x / search_size

                            x_t += displace_x
                            y_t += displace_y

                            x_t = round(max(w//2,x_t))
                            x_t = round(min(img_size-1-w//2, x_t))
                            y_t = round(max(h//2,y_t))
                            y_t = round(min(img_size-1-h//2, y_t))

                            
                            fin.write(','.join([str(x_t-w//2),str(y_t-h//2),str(w),str(h)]))
                            fin.write('\n')


                            # 计算时间
                            time_end = time.time()
                            print("fps：%f",1 / (time_end - time_start))
                            count_flag += 1

                        # if str(tar) == '24':
                        #     print(gt[0].name)
                        #     print(img)
                        if str(gt[0].name) == img: 
                        # 图片路径
                            # print("tar",tar)
                            # print("video",v_path)
                            template = np.array(cv2.imread(img_path, 1), dtype=np.uint8)
                            avg_chans = np.mean(template, axis=(0, 1)) # 3通道平均值
                            # 从左上角坐标转到中心
                            x_t = int(gt[0].iloc[0] + gt[0].iloc[2] // 2) 
                            y_t = int(gt[0].iloc[1] + gt[0].iloc[3] // 2)
                            w, h = gt[0].iloc[2],gt[0].iloc[3]
                            fin.write(','.join([str(x_t-w//2),str(y_t-h//2),str(w),str(h)]))
                            fin.write('\n')

                            wc_z = w + 0.5 * (w + h)
                            hc_z = h + 0.5 * (w + h)
                            s_z = round(np.sqrt(wc_z * hc_z))
                            scale_z = template_size / s_z

                            z_crop = get_subwindow_tracking(template, [x_t, y_t],template_size, s_z, avg_chans)
                            template = Variable(z_crop.unsqueeze(0)).cuda()
                            
                            track_flag = 1
                            count_flag += 1
            else:
                continue
                # break



