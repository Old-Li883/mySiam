from turtle import distance
from urllib import response
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
import math


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
    val_video_path = [] # ????????????video???????????????
    val_video = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            val_video_path.append(os.path.join(root, d))
            val_video.append(d)
        break
    
    for v_path in val_video_path:
        target_gt  = load_gt(v_path)
        dataset[v_path] = target_gt # ??????gt??????

    return dataset


def get_region(offset, img_size, img, x, y):
    if x <= offset <= y and y + offset < img_size:
        img_region = img[y - offset:y + offset + 1, 0:2 * x + 1]
    elif x <= offset <= y and y + offset >= img_size:
        img_region = img[2 * y - img_size:img_size, 0:2 * x + 1]
    elif y <= offset <= x and x + offset < img_size:
        img_region = img[0:2 * y + 1, x - offset:x + offset + 1]
    elif y <= offset <= x and x + offset >= img_size:
        img_region = img[0:2 * y + 1, 2 * x - img_size:img_size]
    elif y + offset >= img_size and x + offset >= img_size:
        img_region = img[2 * y - img_size:img_size, 2 * x - img_size:img_size]
    elif x < offset and y < offset:
        img_region = img[0:2 * y + 1, 0:2 * x + 1]
    else:
        img_region = img[y - offset:y + offset + 1, x - offset:x + offset + 1]

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
    ???pos?????????model_sz????????????????????????????????????????????????????????????
    im:?????????????????????
    model_sz:???????????????????????????
    original_sz:???????????????????????????????????????
    """
    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2

    # ?????????im??????????????????coriginal_sz????????????padding???
    # ?????????????????????original_sz/2????????????????????????????????????????????????????????????????????????????????????????????????padding?????????
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    # ???????????????????????????
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    # ?????????????????????
    if any([top_pad, bottom_pad, left_pad, right_pad]): # padding?????????0?????????????????????
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im # ??????????????????
        # ????????????
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else: # ?????????????????????
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    # ???????????????model_size
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    return im_to_torch(im_patch.copy()) if out_mode in 'torch' else im_patch

def MTA(n, p_old_1, p_old_2, delta):
    delta = (p_old_1-p_old_2)/n + (n-1)*delta/n
    p_next = p_old_1 + delta
    return p_next, delta # delta???????????????


def distance(p_1, p_2):
    return math.sqrt((p_1[0] - p_2[0])**2 + (p_1[1] - p_2[1])**2)




# ???????????????
# kf = cv2.KalmanFilter(4, 2)
# kn = 0
# kf_flag = False
# t_an = 30
# delta = [0, 0]
# # p_old???p_cur???delta?????????????????????
# def ME_algorithm(n, p_old_1, p_old_2, p_cur, kf, kf_flag, kn, delta):
#     # ??????????????????????????????????????????cur?????????????????????????????????kalm???????????????kf_flag???kn??????delta??????
#     if n == 1:
#         kn = 0
#         kf_flag = False
#         delta = [0, 0]

#         # ????????????????????????
#         kf = cv2.KalmanFilter(4,2) # ?????????????????????
#         kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
#         kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
#         kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.003
#         kf.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
#         return False, None, kf, kf_flag, kn, delta

#     elif n < t_an: # ?????????????????????MTA??????
        
#         p_estimate, delta = MTA(n, p_old_1, p_old_2, delta) # ???????????????delta
#         if p_cur: # ?????????p_cur????????????p_cur??????kf
#             p_n = np.array([[np.float32(p_cur[0])],[np.float32(p_cur[1])]])
#         else: 
#             p_n = np.array([[np.float32(p_estimate[0])],[np.float32(p_estimate[1])]])

#         kf.correct(p_n) # ??????????????????????????????????????????
#         predict = kf.predict()
#         return False, p_estimate, kf, kf_flag, kn, delta

#     else:
#         if kf_flag == False:
#             p_estimate, delta = MTA(n, p_old_1, p_old_2, delta)


#             # ??????kf
#             if p_cur: # ?????????p_cur????????????p_cur
#                 p_n = np.array([[np.float32(p_cur[0])], [np.float32(p_cur[1])]])
#             else: # ?????????????????????p_estimate??????
#                 p_n = np.array([[np.float32(p_estimate[0])], [np.float32(p_estimate[1])]])

#             kf.correct(p_n) # ???????????????????????????

#             # ??????????????????
#             p_kalm = kf.predict()
#             if not p_cur:
#                 p_cur = p_estimate
#             if distance(p_kalm, p_cur) < 4:
#                 kn += 1
#                 if kn >= 4:
#                     kf_flag = True
#             else:
#                 kn = 0

#             return False, p_estimate, kf, kf_flag, kn, delta
#         else:
#             p_estimate = kf.predict()
#             if p_cur:
#                 p_n = np.array([[np.float32(p_cur[0])], [np.float32(p_cur[1])]])
#             else:
#                 p_n = np.array([[np.float32(p_estimate[0])], [np.float32(p_estimate[1])]])
#             kf.correct(p_n)
#             return True, p_estimate, kf, kf_flag, kn, delta


if __name__ == '__main__':
    template_size = 25  # ????????????
    template_offset = int(template_size / 2)
    search_size = 65  # ??????????????????
    stride = 1
    search_offset = int(search_size / 2)
    center = search_offset + 1
    img_size = 1024  # ????????????
    ignore_size = 16
    update_fre = 100
    displace_th = 32
    score_size = (search_size - template_size) / stride + 1
    w_influence = 0.35


    # ????????????
    backbone = TestNet()

    # run
    # args = parse_args()
    # net = models.__dict__[
    #     config.SIAMISO.TRAIN.MODEL](embedding_net=backbone) 
    # net = load_pretrain(net, args.resume)

    # test
    net = models.__dict__[
        "SiameseISO"](embedding_net=backbone) 
    net = load_pretrain(net, "./snapshot_iso/checkpoint_e47.pth")



    net.eval() # ?????????batch normalization???dropout
    net = net.cuda()

    # ????????????
    dataset = load_dataset("/mnt3/lichenhao/VISO/validation data/")
    video_keys = list(dataset.keys()).copy()

    transform = transforms.ToTensor()

    # data_path = './infrared_small_object_data/val'

    # ?????????
    window = np.outer(np.hanning(int(score_size)),
                            np.hanning(int(score_size)))
    window /= window.sum()

    result_path = './result_iso_noNeg' # test
    # result_path = args.result_path # run
    if not exists(result_path):
        os.makedirs(result_path)
    
    # train_epoch_name = args.resume.split("/")[-1] # run
    train_epoch_name = "47" # test

    result_path = os.path.join(str(result_path),str(train_epoch_name))
    if not exists(result_path):
        os.makedirs(result_path)

    #???????????????test
    nowtime = time.time()
    result_path = os.path.join(result_path, str(nowtime))
    if not exists(result_path):
        os.makedirs(result_path)

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

        peak_value_txt = result_path + '/' + 'peak_value.txt'
        pv_txt = open(peak_value_txt, 'w')
        for tar, gt in video_target_frame.items():

            track_flag = 0
            result_tar_path = result_path +'/' +str(v_name) + '_'+ str(tar)

            pv_txt.write(str(v_name) + ":" + str(tar) +"\n")

            if not exists(result_tar_path):
                os.makedirs(result_tar_path)
                img_path = os.path.join(result_tar_path,'img')
                if not exists(img_path):
                    os.makedirs(img_path)

            
            result_tar_path_txt = result_tar_path + '/groundtruth_rect.txt'
            with open(result_tar_path_txt, "w") as fin:
            # img????????????
                imgs_path = sorted(os.listdir(videos_path))
                count_flag = 0
                for img in imgs_path:

                    # ????????????
                    if count_flag >= gt[1]:
                        break

                    img_path = os.path.join(videos_path,img)

                    # ??????????????????
                    if track_flag != 0:
                        # ??????????????????
                        count_flag += 1

                        # ????????????
                        time_start = time.time()
                        
                        o_search = np.array(cv2.imread(img_path, 1), dtype=np.uint8)
                        avg_chans = np.mean(o_search, axis=(0, 1)) # 3???????????????
                        d_search = (search_size - template_size) / 2
                        pad = d_search / scale_z
                        s_x = s_z + 2 * pad
                        
                        # TODO??????????????????kalm????????????????????????kalm???????????????search_map???????????????
                        if iskalman_work:
                            x_crop = get_subwindow_tracking(o_search, [predict[0][0], predict[1][0]], search_size, s_x, avg_chans)
                        else:
                            x_crop = get_subwindow_tracking(o_search, [x_t, y_t], search_size, s_x, avg_chans)
                        search = x_crop.unsqueeze(0).cuda()                     


                        # ????????????
                        output = net(template,search)

                        # ??????score_map
                        output = torch.squeeze(output, 0)
                        output = torch.squeeze(output, 0)


                        # ???????????????
                        prediction = output.cpu().detach().numpy()
                        prediction -= np.min(prediction)
                        prediction /= prediction.sum()

                        # ?????????
                        prediction = (1 - w_influence) * prediction + w_influence * window
                        # ???????????????????????????
                        position = np.unravel_index(np.argmax(prediction),
                                                    prediction.shape)

                        # TODO: ???????????????????????????????????????
                        peak_value = prediction[position[0]][position[1]]
                        # print(peak_value)
                        pv_txt.write(str(peak_value)+'\n')

                        # ???????????????
                        # ????????????
                        displace_x = (position[1] - prediction.shape[1] // 2) * s_x / search_size
                        displace_y = (position[0] - prediction.shape[0] // 2) * s_x / search_size

                        if iskalman_work:
                            x_t = predict[0][0] + displace_x
                            y_t = predict[1][0] + displace_y
                        else:
                            x_t += displace_x
                            y_t += displace_y

                        x_t = round(max(0,x_t))
                        x_t = round(min(img_size-1, x_t))
                        y_t = round(max(0,y_t))
                        y_t = round(min(img_size-1, y_t))

                        if iskalman_work:
                            kalm_bbox = [predict[0][0], predict[1][0], w, h] # ???????????????kalm?????????????????????????????????

                            if peak_value > occlusion_threshold:
                                # ????????????
                                trace_array.append((x_t, y_t))
                                kf.correct(np.array([[np.float32(x_t)],[np.float32(y_t)]])) # ??????kalm??????
                            else:
                                # ???????????????????????????????????????kalm??????
                                x_t = kalm_bbox[0]
                                y_t = kalm_bbox[1]
                            predict = kf.predict() # kalm???????????????????????????
                        else:
                            if len(trace_array) > 4:
                                dx = 0
                                dy = 0

                                # ?????????4?????????????????????????????????????????????????????????????????????????????????
                                for i in range(-5, -1):
                                    dx += trace_array[i + 1][0] - trace_array[i][0]
                                    dy += trace_array[i + 1][1] - trace_array[i][1]
                                pre_bbox = [
                                    x_t + dx / 4, y_t + dy / 4,
                                    w, h
                                ]

                                if peak_value < 0.001: # ??????????????????
                                    x_t = pre_bbox[0]
                                    y_t = pre_bbox[1]
                                    isocclution = True
                                else:
                                    if isocclution: # ?????????????????????????????????????????????????????????
                                        if occlution_index != 0: # ????????????????????????
                                            tem_trace_array.append((x_t, y_t))
                                        occlution_index += 1

                                        if occlution_index == 6:
                                            trace_array.extend(tem_trace_array)
                                            isocclution == False

                                    else:
                                        trace_array.append((x_t, y_t))

                            else:
                                trace_array.append((x_t, y_t))

                            # ?????????predict??????????????????kalm????????????????????????????????????
                            # ??????kalm??????????????????????????????????????????
                            if (abs(predict[0][0] - x_t) < 2) and (abs(predict[1][0] - y_t) < 2):
                                kalman_work_num += 1
                                if kalman_work_num == 3:
                                    iskalman_work = True
                                    kalman_work_num = 0
                            else:
                                kalman_work_num = 0

                            kf.correct(np.array([[np.float32(x_t)],[np.float32(y_t)]])) # ??????kalm??????
                            predict = kf.predict() # kalm???????????????????????????

                        x_t = round(max(0,x_t))
                        x_t = round(min(img_size-1, x_t))
                        y_t = round(max(0,y_t))
                        y_t = round(min(img_size-1, y_t))
                        
                        fin.write(','.join([str(x_t-w//2),str(y_t-h//2),str(w),str(h)]))
                        fin.write('\n')


                        # ????????????
                        time_end = time.time()
                        print("fps???%f",1 / (time_end - time_start))

                    if str(gt[0].name) == img: 

                        # ??????????????????
                        count_flag += 1

                        # ????????????
                        template = np.array(cv2.imread(img_path, 1), dtype=np.uint8)
                        avg_chans = np.mean(template, axis=(0, 1)) # 3???????????????
                        # ??????????????????????????????
                        x_t = int(gt[0].iloc[0] + gt[0].iloc[2] // 2) 
                        y_t = int(gt[0].iloc[1] + gt[0].iloc[3] // 2)
                        w, h = gt[0].iloc[2], gt[0].iloc[3]
                        fin.write(','.join([str(x_t-w//2),str(y_t-h//2),str(w),str(h)]))
                        fin.write('\n')

                        wc_z = w + 0.5 * (w + h)
                        hc_z = h + 0.5 * (w + h)
                        s_z = round(np.sqrt(wc_z * hc_z))
                        scale_z = template_size / s_z

                        z_crop = get_subwindow_tracking(template, [x_t, y_t],template_size, s_z, avg_chans)
                        template = Variable(z_crop.unsqueeze(0)).cuda()
                        
                        track_flag = 1
                        
                        # ????????????????????????
                        kf = cv2.KalmanFilter(4,2) # ?????????????????????
                        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
                        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
                        kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.003
                        kf.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1

                        trace_array = [] # ??????????????????
                        tem_trace_array = []
                        predict = [[0], [0]] # kalm???????????????
                        iskalman_work = False # kalm??????????????????
                        kalman_work_num = 0 # kalm?????????
                        occlution_index = 0 # ????????????
                        occlusion_threshold = 0.0015 # ????????????
                        isocclution = False

                        trace_array.append((x_t, y_t))
                        kf.correct(np.array([[np.float32(x_t)],[np.float32(y_t)]]))

    pv_txt.close()
