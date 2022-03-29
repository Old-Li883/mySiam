from calendar import EPOCH
from genericpath import exists
import numpy as np
import pandas as pd
from numba import jit
import argparse
import os
from statistics import mean


def overlap_ratio(rect1, rect2):
    '''Compute overlap ratio between two rects
    Args
        rect:2d array of N x [x,y,w,h]
    Return:
        iou
    '''
    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou


def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    mask = np.sum(gt_bb > 0, axis=1) == 4
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success

def success_error(gt_center, result_center, thresholds, n_frame):
    # n_frame = len(gt_center)
    success = np.zeros(len(thresholds))
    dist = np.ones(len(gt_center)) * (-1)
    mask = np.sum(gt_center > 0, axis=1) == 2
    dist[mask] = np.sqrt(np.sum(
        np.power(gt_center[mask] - result_center[mask], 2), axis=1))
    for i in range(len(thresholds)):
        success[i] = np.sum(dist <= thresholds[i]) / float(n_frame)
    return success


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                        (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamISO Tracking eval')
    parser.add_argument('--epoch', dest='epoch', default='1', help='which epoch model to eval')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # 每一个epoch的所有数据都要和gt中的比一下
    
    pre_path = '/mnt3/lichenhao/SiamDW-master/result_iso'
    val_path = '/mnt3/lichenhao/ICPR 2022_SOT/validation data'
    result_path = './evl_iso' # 这个是存储参数，直接运行这个函数

    # args = parse_args()
    # epoch_num = args.epoch
    # epoch_file_name = 'checkpoint_e{0}.pth'.format(str(epoch_num)) # 手动改这个参数就好
    epoch_file_name = '1648139796.7981474/checkpoint_e47.pth'

    epoch_num = 47
    pre_dir = os.path.join(pre_path,epoch_file_name)
    # epoch_result_path = os.path.join(result_path, str(epoch_num))# 这个是存储参数，直接运行这个函数
    epoch_result_path = '/mnt3/lichenhao/SiamDW-master/evl_iso/1648139796.7981474/'
    if not exists(epoch_result_path):
        os.makedirs(epoch_result_path)

    files = os.listdir(val_path)
    # video的绝对路径，但一个video下有多个目标，多个txt文件
    gt_path = []
    pre_path = []

    # TODO：在这里一起处理了，不加入一个个处理了
    precision_ret_ = {}
    overlap_ret_ = {}


    total_result_txt = epoch_result_path + '/total.txt'
    tf = open(total_result_txt, 'w')
    total_dis = []
    total_over = []

    best = {}
    for f in files:
        video_name = f
        precision_ret_[video_name] = {}
        overlap_ret_[video_name] = {}

        gt_dir = os.path.join(val_path, f)
        gt_path= os.path.join(gt_dir, 'sot')
        # pre_path = os.path.join(pre_dir,f)

        success_error_path = os.path.join(epoch_result_path,'dis')
        if not exists(success_error_path):
            os.makedirs(success_error_path)

        success_overlap_path = os.path.join(epoch_result_path,'over')
        if not exists(success_overlap_path):
            os.makedirs(success_overlap_path)

        error_result_txt = success_error_path + '/' + str(video_name) + '.txt'
        over_result_txt = success_overlap_path + '/' + str(video_name) + '.txt'
        sf = open(error_result_txt, 'w')
        of = open(over_result_txt, 'w')


        # pre_targets_txt = os.listdir(pre_path)
        t_path = sorted(os.listdir(gt_path))

        video_dis = []
        video_over = []
        for target_txt in t_path:
            target_index = target_txt.split(".")[0] # 这里是全名包含了目标起始、结束帧
            
            gt_full_path = os.path.join(gt_path,target_txt)
            gt = pd.read_table(gt_full_path,sep=',',header=None)
            gt.columns = ['g_x','g_y','g_w','g_h']
            gt['g_x'] = gt['g_x'] + gt['g_w'] // 2
            gt['g_y'] = gt['g_y'] + gt['g_h'] // 2

            try:
                target_i = target_index.split("_")[0]
                pre_full_path = pre_dir + '/' + video_name + '_' + target_i + '/groundtruth_rect.txt'
                pre = pd.read_table(pre_full_path,sep=',',header=None)
                pre.columns = ['p_x','p_y','p_w','p_h']

                pre['p_x'] = pre['p_x'] + pre['p_w'] // 2
                pre['p_y'] = pre['p_y'] + pre['p_h'] // 2
            except:
                
                print('00000000000000',pre_full_path)
                continue

            total = pd.merge(gt,pre,left_index=True,right_index=True)
            total = total.sort_index()

            total = total[(total['g_x']!=0) | (total['g_y']!=0) | (total['g_w']!=0) | (total['g_h']==0)]
            if len(gt) != len(pre):
                print(len(gt),len(pre))
                print(pre_full_path)


            gt_center = total[['g_x', 'g_y']].to_numpy()
            tracker_center = total[['p_x', 'p_y']].to_numpy()
            thresholds = np.arange(0, 51, 1)
            if len(gt_center) != 0:
                precision_ret_[video_name][target_index] = success_error(gt_center, tracker_center,thresholds, len(gt_center)) # 加1是加上第一帧
                sf.write('target {0},success_error: {1}'.format(str(target_index), precision_ret_[video_name][target_index]))
                sf.write('\n')
                sf.write('target {0},mean_success_error: {1}'.format(str(target_index), precision_ret_[video_name][target_index].mean()))
                sf.write('\n')
                sf.write('\n')
                video_dis.append(precision_ret_[video_name][target_index].mean())
                total_dis.append(precision_ret_[video_name][target_index].mean())

            gt_traj = total[['g_x', 'g_y', 'g_w', 'g_h']].to_numpy()
            tracker_traj = total[['p_x', 'p_y', 'p_w', 'p_h']].to_numpy()
            if len(gt_traj) != 0:
                overlap_ret_[video_name][target_index] = success_overlap(gt_traj, tracker_traj, len(gt_traj))
                of.write('target {0},success_overlap: {1}'.format(str(target_index), overlap_ret_[video_name][target_index]))
                of.write('\n')
                of.write('target {0},mean_success_overlap: {1}'.format(str(target_index), overlap_ret_[video_name][target_index].mean()))
                of.write('\n')
                of.write('\n')

                video_over.append(overlap_ret_[video_name][target_index].mean())
                total_over.append(overlap_ret_[video_name][target_index].mean())

        sf.write("epoch {0},video {1},mean_success_error is {2}".format(epoch_num,video_name,mean(video_dis)))
        of.write("epoch {0},video {1},mean_success_error is {2}".format(epoch_num,video_name,mean(video_over)))
        sf.close()
        of.close()

    tf.write("epoch {0},mean_success_error is {1}".format(epoch_num,mean(total_dis)))
    tf.write("\n")
    tf.write("epoch {0},mean_success_overlap is {1}".format(epoch_num,mean(total_over)))
    tf.close()

    print('{0} epoch mean video_dis is {1}'.format(str(epoch_num), mean(total_dis)))
    print('{0} epoch mean video_over is {1}\n'.format(str(epoch_num), mean(total_over)))




    
            










