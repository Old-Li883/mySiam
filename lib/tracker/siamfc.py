# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and  Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# siamfc class
# ------------------------------------------------------------------------------
import cv2
import numpy as np

from torch.autograd import Variable
from utils.utils import load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid


class SiamFC(object):
    def __init__(self, info):
        super(SiamFC, self).__init__()
        self.info = info   # model and benchmark info，info包含网络结构信息，数据集信息，以及测试时是不是每个epoch都测试

    def init(self, im, target_pos, target_sz, model, hp=None):
        state = dict()
        # epoch test，读取相关配置数据
        p = FCConfig()

        # single test，不用每个epoch都测试
        if not hp and not self.info.epoch_test:
            prefix = [x for x in ['OTB', 'VOT'] if x in self.info.dataset]
            cfg = load_yaml('./experiments/test/{0}/{1}.yaml'.format(prefix[0], self.info.arch))
            cfg_benchmark = cfg[self.info.dataset]
            p.update(cfg_benchmark)
            p.renew()

        # param tune
        if hp:
            p.update(hp)
            p.renew()


        net = model # 训练好的模型

        avg_chans = np.mean(im, axis=(0, 1)) # 3通道平均值

        # 制作模板时将模板图放大填充padding，再resize
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z)) # 模板图缩放放后的（变成正方形）的边长

        # 将模板图缩放的缩放系数
        scale_z = p.exemplar_size / s_z

        # 制作模板图像，填充均值
        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

        # 固定大小后模板图（127*127）相对于搜索区域（256*256）的padding（理解：就是127*127到256*256一边要填充多少，为后面多尺度做准备）
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z # 放缩后模板对应的padding
        s_x = s_z + 2 * pad # 加上padding后模板的大小

        # 多尺度的两个极值
        min_s_x = 0.2 * s_x
        max_s_x = 5 * s_x
        
        s_x_serise = {'s_x': s_x, 'min_s_x': min_s_x, 'max_s_x': max_s_x}
        p.update(s_x_serise)

        # 增加一个维度（应该是模拟batchSize中的batch）
        z = Variable(z_crop.unsqueeze(0))

        net.template(z.cuda()) # 模板图提取特征放在.zf中

        if p.windowing == 'cosine': # 生成余弦矩阵
            window = np.outer(np.hanning(int(p.score_size) * int(p.response_up)),
                              np.hanning(int(p.score_size) * int(p.response_up)))
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size) * int(p.response_up), int(p.score_size) * int(p.response_up))
        window /= window.sum()

        p.scales = p.scale_step ** (range(p.num_scale) - np.ceil(p.num_scale // 2))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        return state

    def update(self, net, s_x, x_crops, target_pos, window, p):
        # refer to original SiamFC code，跟踪目标得到相应图
        response_map = net.track(x_crops).squeeze().permute(1, 2, 0).cpu().data.numpy()

        # 插值扩大score_map的倍数
        up_size = p.response_up * response_map.shape[0]
        response_map_up = cv2.resize(response_map, (up_size, up_size), interpolation=cv2.INTER_CUBIC)
        temp_max = np.max(response_map_up, axis=(0, 1)) # 找出个面的最大值
        s_penaltys = np.array([p.scale_penalty, 1., p.scale_penalty]) # 乘系数
        temp_max *= s_penaltys
        best_scale = np.argmax(temp_max) # 具有最大值的通道索引,这个索引代表最适合的尺度

        # 找出相应的响应层相应归一化
        response_map = response_map_up[..., best_scale]
        response_map = response_map - response_map.min()
        response_map = response_map / response_map.sum()

        # apply windowing，加上余弦窗后的响应值，因为认为目标不会移动太大，给中心一点补偿
        response_map = (1 - p.w_influence) * response_map + p.w_influence * window
        r_max, c_max = np.unravel_index(response_map.argmax(), response_map.shape) # 响应值最大的位置
        p_corr = [c_max, r_max] # 在score_map放大后的位置

        disp_instance_final = p_corr - np.ceil(p.score_size * p.response_up / 2) # 减二分之一（中心位置），计算得在score_map中的偏移
        disp_instance_input = disp_instance_final * p.total_stride / p.response_up # 在搜索图中的偏移（256*256图）
        disp_instance_frame = disp_instance_input * s_x / p.instance_size # 在原图中的偏移

        new_target_pos = target_pos + disp_instance_frame # 前一帧目标位置加上这一帧偏移得这一帧目标位置

        return new_target_pos, best_scale

    def track(self, state, im):
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        # p.s_x是缩放后加了padding模板的大小
        scaled_instance = p.s_x * p.scales # 多尺度采样系数
        scaled_target = [[target_sz[0] * p.scales], [target_sz[1] * p.scales]] # 多尺度目标对应位置

        # 在多尺度中，以目标位置为中心（目标位置会随尺度变换而变换），抠出p.instance_size大小的搜索区域，但尺度变化
        # 做多尺度的目的是考虑到在跟踪时目标会形变，此时要将目标放缩到与模板相似的尺度
        x_crops = Variable(make_scale_pyramid(im, target_pos, scaled_instance, p.instance_size, avg_chans))

        # 
        target_pos, new_scale = self.update(net, p.s_x, x_crops.cuda(), target_pos, window, p)

        # scale damping and saturation
        p.s_x = max(p.min_s_x, min(p.max_s_x, (1 - p.scale_lr) * p.s_x + p.scale_lr * scaled_instance[new_scale]))


        # 将目标位置调整为多尺度图中目标位置
        target_sz = [(1 - p.scale_lr) * target_sz[0] + p.scale_lr * scaled_target[0][0][new_scale],
                     (1 - p.scale_lr) * target_sz[1] + p.scale_lr * scaled_target[1][0][new_scale]]

        # 边界控制
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

        # 更新参数
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p

        return state


class FCConfig(object):
    # These are the default hyper-params for SiamFC
    # 生产层金字塔的参数
    num_scale = 3 # 层数
    scale_step = 1.0375
    scale_penalty = 0.9745
    scale_lr = 0.590
    
    # 响应值上采样参数
    response_up = 16

    # 余弦窗参数
    windowing = 'cosine'
    w_influence = 0.350

    exemplar_size = 127 # 模板大小
    instance_size = 255 # 搜索区域大小
    score_size = 17 # 分数图大小
    total_stride = 8 # 下采样倍数
    context_amount = 0.5 # 模板缩放比例

    def update(self, newparam=None):
        # 更新属性
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.exemplar_size = self.instance_size - 128
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1
