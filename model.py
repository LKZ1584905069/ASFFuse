import torch.nn as nn
from layers import ConvLeakyRelu2d
from conv_select import Select
import torch
from loss import grad
from t2t_vit import Spatial, Channel
from options import TrainOptions
import kornia.filters as KF
import numpy as np
import cv2

opt = TrainOptions().parse()

# 每个卷积的输出固定值,初始值设为60
conv_out = opt.Conv_out
# 最优卷积比例，和最优卷积个数
good_rate = opt.Conv_goodrate
good_num = int(conv_out * good_rate)
# 最差卷积比例，和最差卷积个数 （先不用）
bad_rate = opt.Conv_badrate
bad_num = int(conv_out * bad_rate)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
    使用两个卷积做成一对
'''


# vis 和 ir 的卷积是交替进行的
# vis 和 ir 的卷积是交替进行的
class Encoder(nn.Module):
    # input_channel是初始图像的维度，默认为 1
    def __init__(self, in_channel=1):
        super(Encoder, self).__init__()
        # 第n个卷积的输入为：conv_out + good_num * n ； 输出为固定的：conv_out
        # vis 图片
        self.vis1 = nn.Sequential(ConvLeakyRelu2d(in_channel, conv_out, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))
        self.vis2 = nn.Sequential(ConvLeakyRelu2d(conv_out + good_num * 1, conv_out, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))
        self.vis3 = nn.Sequential(ConvLeakyRelu2d(conv_out + good_num * 1, conv_out, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))
        self.vis4 = nn.Sequential(ConvLeakyRelu2d(conv_out + good_num * 1, conv_out, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))
        self.vis5 = nn.Sequential(ConvLeakyRelu2d(conv_out + good_num * 1, conv_out, norm='Batch', activation='ReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))

        # ir 图片
        self.ir1 = nn.Sequential(ConvLeakyRelu2d(in_channel, conv_out, norm='Batch', activation='ReLU'),
                                 ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))
        self.ir2 = nn.Sequential(ConvLeakyRelu2d(conv_out + good_num * 1, conv_out, norm='Batch', activation='ReLU'),
                                 ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))
        self.ir3 = nn.Sequential(ConvLeakyRelu2d(conv_out + good_num * 1, conv_out, norm='Batch', activation='ReLU'),
                                 ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))
        self.ir4 = nn.Sequential(ConvLeakyRelu2d(conv_out + good_num * 1, conv_out, norm='Batch', activation='ReLU'),
                                 ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))
        self.ir5 = nn.Sequential(ConvLeakyRelu2d(conv_out + good_num * 1, conv_out, norm='Batch', activation='ReLU'),
                                 ConvLeakyRelu2d(conv_out, conv_out, activation='ReLU'))

    # 因为 vis 和 ir 的卷积是交替进行的，所以要传入两张图片
    def forward(self, vis, ir):
        # vis,ir [16,1,128,128]
        # 源图像,复制 conv_out - good_num 份，用于卷积的输入
        self.source_vi = vis.repeat(1, conv_out - good_num, 1, 1).to(device)
        self.source_ir = ir.repeat(1, conv_out - good_num, 1, 1).to(device)

        # 卷积选择对象
        conv_select2 = Select(vi=vis, ir=ir).conv_select

        vis1 = self.vis1(vis)
        ir1 = self.ir1(ir)
        # good_vis1 = conv_select(vis1)
        # good_ir1 = conv_select(ir1)
        good_vis1 = conv_select2(vis1,sources=vis)
        good_ir1 = conv_select2(ir1,sources=ir)

        vis2 = self.vis2(torch.cat([self.source_vi, good_vis1, good_ir1], 1))
        ir2 = self.ir2(torch.cat([self.source_ir, good_vis1, good_ir1], 1))
        # good_vis2 = conv_select(vis2)
        # good_ir2 = conv_select(ir2)
        good_vis2 = conv_select2(vis2,sources=vis)
        good_ir2 = conv_select2(ir2,sources=ir)

        vis3 = self.vis3(torch.cat([self.source_vi, good_vis2, good_ir2], 1))
        ir3 = self.ir3(torch.cat([self.source_ir, good_vis2, good_ir2], 1))
        # good_vis3 = conv_select(vis3)
        # good_ir3 = conv_select(ir3)
        good_vis3 = conv_select2(vis3,sources=vis)
        good_ir3 = conv_select2(ir3,sources=ir)

        vis4 = self.vis4(torch.cat([self.source_vi, good_vis3, good_ir3], 1))
        ir4 = self.ir4(torch.cat([self.source_ir, good_vis3, good_ir3], 1))
        # good_vis4 = conv_select(vis4)
        # good_ir4 = conv_select(ir4)
        good_vis4 = conv_select2(vis4,sources=vis)
        good_ir4 = conv_select2(ir4,sources=ir)

        vis5 = self.vis5(torch.cat([self.source_vi, good_vis4, good_ir4], 1))
        ir5 = self.ir5(torch.cat([self.source_ir, good_vis4, good_ir4], 1))
        # good_vis5 = conv_select(vis5)
        # good_ir5 = conv_select(ir5)
        good_vis5 = conv_select2(vis5,sources=vis)
        good_ir5 = conv_select2(ir5,sources=ir)

        return torch.cat([good_vis1, good_ir1, good_vis2, good_ir2, good_vis3, good_ir3, good_vis4, good_ir4, good_vis5, good_ir5],1)


'''
    使用一个梯度算子增强模块的纹理内容
    1.将来自不同分布的特征合并： conv3x3 + ReLU
    2.将channel分成四等份，分别于 vis梯度、ir梯度、sigmod(vis)和sigmod(ir)相乘
    3.相乘后的特征合并： conv3x3 + ReLU + conv3x3 + ReLU
    4.对合并后的特征降维： conv 1x1 + ReLU + BN
    5.对原始特征降维：conv1x1 + ReLU + BN
    6.将增强特征和原始特征串联,压缩到input的一半：
'''


# 输入选择用opt的参数后面好改
# 输出的Channel是输入的一半
class Texture_enhance(nn.Module):
    # def __init__(self, in_channel = int(2*(opt.Conv_out + 5*opt.Conv_out*opt.Conv_goodrate)) ):
    def __init__(self, in_channel=int(10 * opt.Conv_out * opt.Conv_goodrate)):
        super(Texture_enhance, self).__init__()
        # 1.将来自不同分布的特征合并： conv3x3 + ReLU
        self.conv1 = ConvLeakyRelu2d(in_channel, in_channel, norm='Batch', activation='ReLU')
        # 2.将channel分成四等份，分别于 vis梯度、ir梯度、sigmod(vis)和sigmod(ir)相乘
        # 在forwar中进行
        # 3.相乘后的特征合并： conv3x3 + ReLU + conv3x3 + ReLU
        self.conv2 = ConvLeakyRelu2d(in_channel, in_channel, norm='Batch', activation='ReLU')
        self.conv3 = ConvLeakyRelu2d(in_channel, in_channel, norm='Batch', activation='ReLU')
        # 4.对合并后的特征降维： conv 1x1 + ReLU + BN
        self.conv4 = ConvLeakyRelu2d(in_channel, int(in_channel / 2), kernel_size=1, padding=0, norm='Batch',
                                     activation='ReLU')
        # 5.对原始特征降维：conv1x1 + ReLU + BN
        self.conv5 = ConvLeakyRelu2d(in_channel, int(in_channel / 2), kernel_size=1, padding=0, norm='Batch',
                                     activation='ReLU')

        # 6.将增强特征和原始特征串联,压缩到input的一半：conv3x3+ReLU 和 conv1x1 + BN + ReLU
        self.conv6 = nn.Sequential(ConvLeakyRelu2d(in_channel, in_channel, activation='ReLU'),
                                   ConvLeakyRelu2d(in_channel, int(in_channel / 2), kernel_size=1, padding=0,
                                                   norm='Batch', activation='ReLU'))

    def forward(self, x, vi, ir):
        # c 是 channel，用来计算每份channel的个数
        _, c, _, _ = x.shape
        conv1 = self.conv1(x)
        # # 2.将channel分成四等份，分别于 vis梯度、ir梯度、sigmod(vis)和sigmod(ir)相乘
        # c = int(c / 4)
        # fea1 = conv1[:, 0:c, :, :]
        # fea2 = conv1[:, c:c * 2, :, :]
        # fea3 = conv1[:, c * 2:c * 3, :, :]
        # fea4 = conv1[:, c * 3:, :, :]  # 因为通道能不能整除4不一定，所以这里不写死
        #
        # # 乘法
        # # fea1 = fea1 * grad(vi)
        # # fea2 = fea2 - grad(ir)
        # fea1 = fea1 * KF.spatial_gradient(vi, order=2).abs().sum(dim= 2)
        # fea2 = fea2 * KF.spatial_gradient(ir, order=2).abs().sum(dim= 2)
        # fea3 = fea3 * nn.Sigmoid()(vi)
        # fea4 = fea4 * nn.Sigmoid()(ir)
        #
        # merge = torch.cat([fea1, fea2, fea3, fea4], 1)

        c = int(c/3)
        grad_ir = KF.spatial_gradient(ir, order=2).abs().sum(dim= 2)
        grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim= 2)
        grad_joint = torch.max(grad_ir, grad_vi)
        fea1 = conv1[:,0:c,:,:]
        fea2 = conv1[:,c:2*c,:,:]
        fea3 = conv1[:,2*c:,:,:]
        fea1 = fea1 + grad_joint
        fea2 = fea2 * nn.Sigmoid()(vi)
        fea3 = fea3 * nn.Sigmoid()(ir)
        merge = torch.cat([fea1,fea2,fea3],1)

        # 3.相乘后的特征合并： conv3x3 + ReLU + conv3x3 + ReLU
        conv2 = self.conv2(merge)
        conv3 = self.conv3(conv2)
        # 对并和后的增强特征压缩降维
        conv4 = self.conv4(conv3)
        # 将原始特征压缩到Channel的一半
        conv5 = self.conv5(conv1)
        out = self.conv6(torch.cat([conv4, conv5], 1))

        # 做一个可视化
        # savePatH = './saveFea/joint/'
        # for j in range(c*4):
        #     img_fusion = np.array(merge[0,j, :, :].detach().squeeze().cpu() * 255)
        #     img = img_fusion.astype('uint8')
        #     cv2.imwrite(savePatH+'enhance_'+str(j)+'.bmp', img)

        # 特征合并和返回，输出的Channel是输入的一半
        return out


'''
 1.将输入特征图Channel压缩到一半，self.con1  conv3x3 和 conv1x1
 2.将压缩后的特征图进行两次下采样
 3.将下采样的数据送入两个transformer得到权重图
 4.将transformer的结果两次上采样到原来大小
 5.将上采样后的权重图和Channel减半后的特征图相乘并返回
'''


# Channel and Spatial Transformer
# Transformer得到的是一个权重图，需要与原特征相乘，结果已经乘过了
# 输出的Channel是输入的 一半
class Transformer(nn.Module):
    # def __init__(self,in_channel = int(2*(opt.Conv_out + 5*opt.Conv_out*opt.Conv_goodrate))):
    def __init__(self, in_channel=int(10 * opt.Conv_out * opt.Conv_goodrate)):
        super(Transformer, self).__init__()

        # 将通道压缩到原来的一半，使用 conv3x3 和 conv1x1
        self.conv1 = nn.Sequential(ConvLeakyRelu2d(in_channel, in_channel, activation='ReLU', norm='Batch'),
                                   ConvLeakyRelu2d(in_channel, int(in_channel / 2), kernel_size=1, padding=0,
                                                   activation='ReLU'))

        # 因为transformer的计算量大，所以采用下采样
        self.down = nn.AvgPool2d(2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 此处暂时与原论文保持一致，还未改动
        # size是 没有用到，应该是保持和原来的transformer一致，是输入图像的大小
        # embed_dim是 目标向量长度，也是图中对应的 MLP，
        # patch_size 是将图片分为 16 x 16 的小patch
        # channel 是传入的通道数量
        self.channel = Channel(size=32, embed_dim=128, patch_size=16, channel=int(in_channel / 2))
        self.spatial = Spatial(size=256, embed_dim=1024 * 2, patch_size=4, channel=int(in_channel / 2))

    def forward(self, x):
        # 图片大小128*128，下采样两次即可达到TGFusion的大小
        conv1 = self.conv1(x)
        down1 = self.down(conv1)
        down2 = self.down(down1)
        channel = self.channel(down2)
        attr = self.spatial(channel)
        up1 = self.up(attr)
        up2 = self.up(up1)

        # transformer得到的权重图和 conv1 相乘
        out = conv1 * up2

        # 做一个可视化
        # savePatH = './saveFea/'
        #
        # for j in range(out.shape[1]):
        #     img_fusion = np.array(out[0, j, :, :].detach().squeeze().cpu() * 255)
        #     img = img_fusion.astype('uint8')
        #     cv2.imwrite(savePatH + 'trans' + str(j) + '.bmp', img)
        return out


# 解码器的输入是 两个模块的连接
class Decoder(nn.Module):
    # def __init__(self,in_channel = int(2*(opt.Conv_out + 5*opt.Conv_out*opt.Conv_goodrate))):
    def __init__(self, in_channel=int(10 * opt.Conv_out * opt.Conv_goodrate)):
        super(Decoder, self).__init__()
        channels = [in_channel, 128, 64, 32, 1]
        self.conv1 = ConvLeakyRelu2d(channels[0], channels[0], norm='Batch', activation='ReLU')
        self.conv2 = ConvLeakyRelu2d(channels[0], channels[1], norm='Batch', activation='ReLU')
        self.conv3 = ConvLeakyRelu2d(channels[1], channels[2], norm='Batch', activation='ReLU')
        self.conv4 = ConvLeakyRelu2d(channels[2], channels[3], norm='Batch', activation='ReLU')
        self.conv5 = ConvLeakyRelu2d(channels[3], channels[4], activation='Sigmoid')

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv5


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.en = Encoder()
        self.enhence = Texture_enhance()
        self.transfromer = Transformer()
        self.de = Decoder()

    def forward(self, vis, ir):
        en = self.en(vis=vis, ir=ir)
        enhence = self.enhence(en, vi=vis, ir=ir)
        transfromer = self.transfromer(en)
        out = self.de(torch.cat([enhence, transfromer], 1))
        return out





