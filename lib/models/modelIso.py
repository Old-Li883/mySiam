import torch
import torch.nn as nn
import torch.nn.functional as fct
from torch import randn, cat, matmul


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_ch,
        #               out_channels=int(in_ch / 2),
        #               kernel_size=1),
        #     nn.Conv2d(in_channels=int(in_ch / 2),
        #               out_channels=out_ch,
        #               kernel_size=3,
        #               padding=1))
        self.conv1 = nn.Conv2d(in_channels=in_ch,
                      out_channels=int(in_ch / 2),
                      kernel_size=1)

        self.conv2 = nn.Conv2d(in_channels=int(in_ch / 2),
                      out_channels=out_ch,
                      kernel_size=3,
                      padding=1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)

        return res


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        # 特征提取网络结构
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3,
        #               out_channels=64,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.1),
        # )

        self.modules1conv = nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        self.modules1bn = nn.BatchNorm2d(64)
        self.modules1relu = nn.LeakyReLU(0.1)

        self.res1 = ResidualBlock(64, 128)

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.1),
        # )

        self.modules2conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.modules2bn = nn.BatchNorm2d(128)
        self.modules2relu = nn.LeakyReLU(0.1)

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.1),
        # )

        self.modules3conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.modules3bn = nn.BatchNorm2d(256)
        self.modules3relu = nn.LeakyReLU(0.1)

        self.res2 = ResidualBlock(256, 128)

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.1),
        # )

        self.modules4conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.modules4bn = nn.BatchNorm2d(128)
        self.modules4relu = nn.LeakyReLU(0.1)

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))

        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.train_num = 0

    def forward(self, x):
        conv1 = self.modules1conv(x)
        conv1 = self.modules1bn(conv1)
        conv1 = self.modules1relu(conv1)

        res1 = self.res1(conv1)

        conv2 = self.modules2conv(conv1)
        conv2 = self.modules2bn(conv2)
        conv2 = self.modules2relu(conv2)

        # print(res1.shape)
        conv2 = cat((conv2, res1), dim=1)

        # conv3 = self.conv3(conv2)
        conv3 = self.modules3conv(conv2)
        conv3 = self.modules3bn(conv2)
        conv3 = self.modules3relu(conv2)

        res2 = self.res2(conv3)

        # conv4 = self.conv4(conv3)
        conv4 = self.modules4conv(conv3)
        conv4 = self.modules4bn(conv4)
        conv4 = self.modules4relu(conv4)
        conv4 = cat((conv4, res2), dim=1)

        output = self.conv5(conv4)

        # output = fct.interpolate(conv5, conv2.shape[3], mode='bilinear', align_corners=False)

        return output

    def unlock(self):
        # 逐层训练
        for p in self.parameters():
            p.requires_grad = False

        for i in range(1, self.train_num):  # zzp pay attention here
            if i <= 5:
                m = self.features.layer2[-i]
            elif i <= 8:
                m = self.features.layer1[-(i - 5)]
            else:
                m = self.features

            for p in m.parameters():
                p.requires_grad = True
        self.eval()
        self.train()

    def unfix(self, ratio):
        """
        unfix gradually as paper said
        """
        eps = 1e-5
        if abs(ratio - 0.0) < eps:
            self.train_num = 2  # epoch0 1*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.1) < eps:
            self.train_num = 3  # epoch5 2*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.2) < eps:
            self.train_num = 4  # epoch10 3*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.3) < eps:
            self.train_num = 6  # epoch15 4*[1,3,1]  stride2pool makes stage2 have a more index
            self.unlock()
            return True
        elif abs(ratio - 0.5) < eps:
            self.train_num = 7  # epoch25 5*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.6) < eps:
            self.train_num = 8  # epoch30 6*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.7) < eps:
            self.train_num = 9  # epoch35 7*[1,3,1]
            self.unlock()
            return True

        return False


class SiameseISO(nn.Module):
    def __init__(self, embedding_net, upscale=False):
        super(SiameseISO, self).__init__()
        self.features = embedding_net
        self.match_BatchNorm = nn.BatchNorm2d(1)
        self.upscale = upscale
        self.non_local_attn = _NonLocalBlock2D(
            in_channels=128)  # 通过定义类的方式定义类定义子网络
        self.cross_attn = CrossLocal(in_channels=128)  # 通过类的方式定义类定义子网络

        self.adjust_attn = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )

        self.adjust_attn_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.adjust_attn_bn = nn.BatchNorm2d(128)
        self.adjust_attn_relu = nn.LeakyReLU(0.1)

    def forward(self, x1, x2):
        # 提取特征模板特征、搜索图特征
        embedding_template = self.features(x1)
        embedding_search = self.features(x2)

        # self-attention
        non_local_attn_template = self.non_local_attn(embedding_template)
        non_local_attn_search = self.non_local_attn(embedding_search)

        # cross-attention
        cross_attn_template = self.cross_attn(embedding_template,
                                              embedding_search)
        cross_attn_search = self.cross_attn(embedding_search,
                                            embedding_template)

        # 模板图attention特征
        attn_template = cat((non_local_attn_template, cross_attn_template),
                            dim=1)
        # 搜索图attention特征
        attn_search = cat((non_local_attn_search, cross_attn_search), dim=1)

        # 调整层
        attn_template = self.adjust_attn(attn_template)
        # attn_template = self.adjust_attn_conv(attn_template)
        # attn_template = self.adjust_attn_bn(attn_template)
        # attn_template = self.adjust_attn_relu(attn_template)


        attn_search = self.adjust_attn(attn_search)
        # attn_search = self.adjust_attn_conv(attn_search)
        # attn_search = self.adjust_attn_bn(attn_search)
        # attn_search = self.adjust_attn_relu(attn_search)

        # 相关层
        match_map = self.match_corr(attn_template, attn_search)

        return match_map

    def match_corr(self, embed_tem, embed_srh):
        b, c, h, w = embed_srh.shape
        need_pad = 0 #int(embed_tem.shape[2] / 2)

        match_map = fct.conv2d(embed_srh.view(1, b * c, h, w),
                               embed_tem,
                               stride=1,
                               padding=need_pad,
                               groups=b)

        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_BatchNorm(match_map)

        if self.upscale:
            match_map = fct.interpolate(match_map,
                                        self.upsc_size,
                                        mode='bilinear',
                                        align_corners=False)

        return match_map

    def conv(self, x):
        x = self.embedding_net(x)

        return x


# 损失函数计算
class BCEWeightLoss(nn.Module):
    def __init__(self):
        super(BCEWeightLoss, self).__init__()

    def forward(self, input, target, weight=None):
        return fct.binary_cross_entropy_with_logits(input,
                                                    target,
                                                    weight,
                                                    reduction='sum')


class _NonLocalBlock2D(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param sub_sample:
        :param bn_layer:
        """
        super(_NonLocalBlock2D, self).__init__()
        self.dimension = 2
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # 通道减半
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:  # 是否需要下采样
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)，序列，通道，序列图数，高，宽
        :param return_nl_map: if True return z, nl_map, else only return z.是否返回non_local_map
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = matmul(theta_x, phi_x)  # 矩阵相乘
        f_div_C = fct.softmax(f, dim=-1)

        y = matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class CrossLocal(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(CrossLocal, self).__init__()

        self.in_channels = in_channels
        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)

        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0), bn(self.in_channels))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, main_feature, cross_feature):
        batch_size = main_feature.size(0)
        main_size = main_feature.size(2)
        cross_size = cross_feature.size(2)

        x = self.g(cross_feature).view(batch_size, self.inter_channels, -1)
        x = x.permute(0, 2, 1)

        y = self.theta(cross_feature).view(batch_size, self.inter_channels, -1)

        # 调整尺寸和cross尺寸一致
        z = fct.interpolate(main_feature,
                            cross_size,
                            mode='bilinear',
                            align_corners=False)

        z = self.phi(z).view(batch_size, self.inter_channels, -1)
        z = z.permute(0, 2, 1)

        # 矩阵相乘
        f = torch.matmul(x, y)
        f_div_C = fct.softmax(f, dim=-1)

        output = torch.matmul(f_div_C, z)
        output = output.permute(0, 2, 1).contiguous()
        output = output.view(batch_size, self.inter_channels,
                             *cross_feature.size()[2:])

        # 1*1卷积调整通道数
        output = self.W(output)
        output = fct.interpolate(output,
                                 main_size,
                                 mode='bilinear',
                                 align_corners=False)
        output += main_feature
        return output


if __name__ == '__main__':
    testnet = TestNet()
    net = SiameseISO(testnet)
    cross = CrossLocal(128)
    non = _NonLocalBlock2D(in_channels=128)

    template = randn(8, 1, 25, 25)
    search = randn(8, 1, 63, 63)
    out = net(template, search)
    # print(out.shape)
