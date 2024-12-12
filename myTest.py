# import torch
#
# from Mymodel.model import Encoder
# import numpy as np
#
# # a = Encoder(1)
#
# # a = [8,2,34,6,12,3]
# # b = a.copy()
# # a.sort()
# # a.reverse()
# # print(a)
# # print(b)
#
# # # 全局平均池化
# # a = np.arange(16*4).reshape(2,2,4,4)
# # print('初始化a：',a)
# # a = torch.from_numpy(a)
# # a = a.float()
# # a = a.mean(-1).mean(-1)
# # print('最终的a',a)
# # print(a)
#
# # a = []
# # a.append(1)
# # a.append(3)
# # print(a)
# # fea = np.arange(16*4).reshape(2,4,2,4)
# # print('------fea----------')
# # print(fea)
# # print('------取值后的fea--------')
# # print(fea[:,a,:,:])
# # print('-------fea 1---------')
# # print(fea[:,1,:,:])
# # print('-------fea 3---------')
# # print(fea[:,3,:,:])
#
# # from functools import partial
# from options import TrainOptions
# a = TrainOptions()
# a = a.parse()
# print(a.Loss_weight)
#
# # 测试dataset数据的读取
# # from dataset import *
# # from options import *
# # opt = TrainOptions().parse()
# # a = dataset(opt)
#
# # from model import Model,Encoder,Texture_enhance,Transformer,Decoder
# # # a= Decoder()
# # a = Model()
# # print(a)
#
import torch

# a = [1.2,21,3.5]
# a.sort()
# print(a)
# b = [2,1,1.2]
#
# print(type(a))
# print(a[0])
# print(b.index(a[0]))
# import torch
# a = torch.tensor([])
# print(type(a))
# print(a.shape)
#
# a = torch.tensor([1])
# b = a.unsqueeze(0)
# print(a)
# print(b)