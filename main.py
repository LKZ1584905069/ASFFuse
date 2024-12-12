import glob
from datetime import datetime
import torch
from options import TrainOptions
from torch.utils.data import DataLoader
from dataset import *
from model import Model
import os
from torch.optim import Adam
from loss import SSIMLoss, final_ssim
from  test import test
import kornia.filters as KF
import torch.nn.functional as F
# import numpy as np
# import cv2
# import kornia.utils as KU


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 权重初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, torch.nn.Conv2d):
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            print('卷积权重初始化失败')
    elif isinstance(m, torch.nn.BatchNorm2d):
        try:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        except:
            print('BN初始化失败')
def dataLoader(opt, dataset):
    return  DataLoader(
            dataset,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=opt.n_workers,
            drop_last=True,
         )

def train(opt, dataset):
    # 加载数据
    dataloader = dataLoader(opt, dataset)
    # 训练轮数
    train_num = len(dataset)
    # model
    print('\n--- load model ---')
    print(f'------ batch_num:{train_num} --------')
    # model = SuperFusion(opts)
    model = Model()
    # 初始化权重，放在cuda前面
    model.apply(gaussian_weights_init)
    model.cuda()

    # 开始计时
    from datetime import datetime
    start_time = datetime.now()
    # count 用于打印数据，count%10==0打印损失函数，count%50=0保存模型一次
    count = 0
    batch_num = len(dataloader)

    # 设置网络优化器
    optimizer = Adam(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
    model.train()
    print('~~~Main 训练开始！~~~~')

    # 训练的轮数， opt.epoch = 1
    for ep in range(opt.epoch):
        for it, (img_ir, img_vi) in enumerate(dataloader):
            count += 1
            print(f'--第{ep}轮---{count} / {batch_num}----  ')

            # 优化器梯度清零
            optimizer.zero_grad()

            img_vi = img_vi.to(device)
            img_ir = img_ir.to(device)

            # 生成的图片名命为 gen_image
            gen_iamge = model(vis = img_vi, ir = img_ir)

            # weight 是 grad、SSIM、Intens 的比例系数
            weights = opt.Loss_weight    # x先默认[1,1,10]后面改

            # 使用Sobel函数求导
            grad_ir = KF.spatial_gradient(img_ir, order=2).abs().sum(dim=[1, 2])
            grad_vi = KF.spatial_gradient(img_vi, order=2).abs().sum(dim=[1, 2])
            grad_fus = KF.spatial_gradient(gen_iamge, order=2).abs().sum(dim=[1, 2])
            grad_joint = torch.max(grad_ir, grad_vi)

            # 第二步：求 vis 和 ir 中用不上的梯度
            zeros = torch.zeros_like(grad_vi).to(device)
            ones = torch.ones_like(grad_vi).to(device)

            vis_dis = torch.where(grad_vi - grad_ir > 0, zeros, ones )
            ir_dis = 1 - vis_dis

            dis_vi = grad_vi * vis_dis  # [b,c,h,w]
            dis_ir = grad_ir * ir_dis  # [b,c,h,w]
            # 第三步：正相关是IF靠近联合梯度，负相关是IF原理用不上的梯度
            d_ap = torch.mean((grad_fus - grad_joint) ** 2)
            d_an_ir = torch.mean((grad_fus - dis_ir) ** 2)
            d_an_vi = torch.mean((grad_fus - dis_vi) ** 2)

            # 第四步：计算ContrastLoss
            loss_grad = d_ap / (d_an_vi + 1e-7) + d_ap / (d_an_ir + 1e-7)

            # SSIM局部损失
            loss_ssim = 1 - final_ssim(img_ir, img_vi, gen_iamge)


            # 强度损失
            loss_intensity = 0.5 * F.l1_loss(gen_iamge,img_ir) + 0.5 * F.l1_loss(gen_iamge, img_vi)

            loss_total =  1 * loss_grad + weights[1] * loss_ssim + weights[2] * loss_intensity


            # total loss 生成器的损失
            loss_total.backward()

            optimizer.step()
            # scheduler.step()

            # 打印损失函数
            if count % 10 == 0:
                elapsed_time = datetime.now() - start_time
                print('loss_grad: %s, loss_ssim: %s, loss_intensity: %s ,loss_total: %s, selapsed_time: %s' % (
                    loss_grad.item(), loss_ssim.item(), loss_intensity.item(), loss_total.item(), elapsed_time))


            # 保存模型参数(仅仅保留参数)，估计是因为这个模型太大了，保存模型和参数的话占用存储太多
            if count % 100 == 0:
                # save model
                model.eval()
                model.cpu()
                save_model_filename = "Epoch_" + str(count) + "_iters_" + str(count) + ".model"
                save_model_path = os.path.join(opt.saveModelPath ,save_model_filename)
                torch.save(model.state_dict(), save_model_path)
                model.train()
                model.cuda()

            #  每个250轮验证一次结果
            if count % 300 == 0:
                # 因为计算机是并行的cpu的写入不能满足代码的调用速度，所以使用本轮前面的保存的一个model
                modelPath = "Epoch_" + str(count-100) + "_iters_" + str(count-100) + ".model"

                test(modelPath)

    # 训练结束，保存最后一个模型
    model.eval()
    model.cpu()
    save_model_filename = "Final_epoch_" + str(count) + ".model"
    # args.save_model_dir = models
    save_model_path = os.path.join(opt.saveModelPath, save_model_filename)
    torch.save(model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)




if __name__ == '__main__':
    opt = TrainOptions().parse()

    is_train = True

    if is_train:
        # 数据加载
        dataset = dataset(opt.traindata)
        # dataset = dataset(opt.testdata)
        train(opt,dataset)

