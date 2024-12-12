import torch
import numpy as np
import math
from options import TrainOptions
import kornia.filters as KF
import torch.nn.functional as F
import cv2
from loss import grad

opt = TrainOptions().parse()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 根据全局平均池化的大小选择特征图
# input 是要计算最优的特征图，n的要选的特征图的数量，默认是18张
def Global_Average_Pooling(input):
    # 全局平均池化选择平均后最大的特征图作为信息最多的特征图
    _,c,_,_ = input.shape
    results = []
    for feas in input:
        gav = feas.mean(-1).mean(-1)
        results.append(gav)
    return results


# 特征图减去源图像的像素
def L1_piexl(input, sources):
    _, c, _, _ = input.shape
    results = []
    # 第一步：input - sources L1损失
    sources = sources.repeat(1,opt.Conv_out,1,1)
    piexl = F.l1_loss(input,sources,reduction='none')
    # 第二步：分batch求每一张特征的差值大小
    for feas in piexl:
        result = torch.zeros(c).cuda()
        index = 0
        for one_fea in feas:
            # one_fea.shape:torch.Size([128, 128])
            result[index] += one_fea.sum()
            index += 1
        results.append(result)
    return results

# 特征图减去源图像的梯度
def L1_grad(input, sources):
    results = []
    # 第一步：对 input 求梯度
    sources = sources.repeat(1, opt.Conv_out, 1, 1)
    grad_input = KF.spatial_gradient(input, order=1).abs().sum(dim=2)
    # 第二步：对 input 和 sources 梯度做 L1损失
    grad = F.l1_loss(grad_input,sources,reduction='none')
    # 第三步：分batch求每一张特征的差值大小
    for feas in grad:
        a = feas.sum(dim=[1,2])
        results.append(a)
    return results



class Select():
    def __init__(self, vi=None, ir=None ):
        self.piexl_mean = (vi + ir) / 2
        grad_ir = grad(ir)
        grad_vi = grad(vi)
        self.grad_joint = torch.max(grad_ir, grad_vi)


    def conv_select(self, input , sources=None, n=int(opt.Conv_out * opt.Conv_goodrate)):

        with torch.no_grad():

            grad = L1_grad(input, self.grad_joint)
            # Gav
            gav = Global_Average_Pooling(input)


        b, c, h, w = input.shape
        # 保存每个 batch 16个图片的最优特征图
        batch_result = torch.zeros([b, n, h, w],requires_grad=False).to(device)  # n 最终要保留的特征图的数量
        idx = 0  # idx 是batch_result对应的batch下标
        for i in range(b):
            # 将选出来的最优特征图反映到 input 中 并且返回最优图 （16，最优个数，H，W）
            # c个channel 全部计算完之后 ,拉到统一格式【0-40】
            # _, piexl_index = piexl[i].sort()
            _, grad_index = grad[i].sort()
            _, gav_index = gav[i].sort()
            # myTest 77-86 拉到统一格式【0-40】
            for j in range(c):
                # piexl[i][piexl_index[j]] = j
                grad[i][grad_index[j]] = j
                # gav 是越大越好，和grad和piexl相反
                gav[i][gav_index[j]] = c - j - 1


            # 每张特征图的得分
            # score = piexl[i] + grad[i]
            score = gav[i] + grad[i]
            # score = grad[i]


            # 排序,
            score_index = score.sort()[1]
            # 打印排序结果
            # print('score_index:排序结果')
            # print(score_index)

            # # 做一个可视化
            # savePatH = './saveFea/'
            # for j in range(c):
            #     img_fusion = np.array(input[idx, score_index[j], :, :].detach().squeeze().cpu() * 255)
            #     # img_fusion = np.array(input[idx, j, :, :].detach().squeeze().cpu() * 255)
            #     img = img_fusion.astype('uint8')
            #     cv2.imwrite(savePatH + 'ir_' + str(j) + '.bmp', img)
            # # 被选择的特征图的可视化保存路径
            # savePatH = './saveFea/choice/'

            # 选取得分最大的 n 个
            for k in range(n):
                batch_result[idx] += input[idx, score_index[k], :, :]
                # # 做一个被选中特征图的可视化
                # img_fusion = np.array(input[idx, score_index[k], :, :].detach().squeeze().cpu() * 255)
                # img = img_fusion.astype('uint8')
                # cv2.imwrite(savePatH + 'ir_' + str(k) + '.bmp', img)
            idx += 1
        return batch_result






