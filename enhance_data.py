# author:xxy,time:2022/2/22
############ tf的预定义 ############
from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import time
import os
from glob import glob
import cv2
from  PIL import Image
from imageio import imsave

############ 常量的预定义 ############
batch_size = 70  # 每张图片剪70张
patch_size_x = 224 # 剪后图片大小
patch_size_y = 224
save_path = './result/' # 剪后图片保存

############ 变量的预定义 ############
sess = tf.InteractiveSession()
# 因为每一块都是随机大小，所以定义为None
vi = tf.placeholder(tf.float32, [None, None, None, 1], name='vi')
ir = tf.placeholder(tf.float32, [None, None, None, 1], name='ir')


############ 准备数据 ############
# load_data
train_ir_data = []
train_vi_data = []

# glob()返回与路径名模式匹配的路径列表。
train_ir_data_names = glob('./data/ir/*') #  实际训练使用
train_vi_data_names = glob('./data/vi/*') #  实际训练使用
train_ir_data_names.sort()
train_vi_data_names.sort()
print('[*] Number of training data_ir/vi: %d' % len(train_ir_data_names))

def load_images(file):
    # Image.open(file)打开并识别给定的图像文件。
    im = Image.open(file)
    # img = np.array(im, dtype="float32") / 255.0
    img = np.array(im, dtype="float32")
    img_norm = np.float32(img)
    return img_norm

def rgb_ycbcr(img_rgb):
    R = tf.expand_dims(img_rgb[:, :, 0], axis=-1)
    G = tf.expand_dims(img_rgb[:, :, 1], axis=-1)
    B = tf.expand_dims(img_rgb[:, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255
    img_ycbcr = tf.concat([Y, Cb, Cr], axis=-1)
    return img_ycbcr

def rgb_ycbcr_np(img_rgb):
    R = np.expand_dims(img_rgb[:, :, 0], axis=-1)
    G = np.expand_dims(img_rgb[:, :, 1], axis=-1)
    B = np.expand_dims(img_rgb[:, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
    img_ycbcr = np.concatenate([Y, Cb, Cr], axis=-1)
    return img_ycbcr


for idx in range(len(train_ir_data_names)):
    # 获取到源图片
    im_before_ir = load_images(train_ir_data_names[idx])
    if im_before_ir.ndim == 3:  # 如果是三通道的就RGB-YUV
        # 转化为灰度图
        ir_gray = cv2.cvtColor(im_before_ir, cv2.COLOR_RGB2GRAY)
        ir_gray = rgb_ycbcr_np(im_before_ir)[:, :, 0]
    else:
        ir_gray = im_before_ir

    # 加入训练数组
    train_ir_data.append(ir_gray)

    im_before_vi = load_images(train_vi_data_names[idx])
    if im_before_vi.ndim == 3:
        vi_gray = cv2.cvtColor(im_before_vi, cv2.COLOR_RGB2GRAY)
        vi_y = rgb_ycbcr_np(im_before_vi)[:,:,0]
    else:
        vi_y = im_before_vi
    train_vi_data.append(vi_y)  # 是归一化之后的图像形成一个list组


epoch = len(train_ir_data_names)  # 总的图片数量，用于剪的时候大循环
train_phase = 'decomposition'
numBatch = len(train_ir_data) // 24  # 批数据量是10,一个小patch图片大小是48 batch_size = 24





############ 训练开始~！ ############
start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0
# 每次步长是一个一张图片
for i in range(epoch):
    h, w = train_ir_data[image_id].shape
    if h <= 224 or w <= 224:
        image_id = (image_id + 1) % len(train_ir_data)
    else:
        for batch_id in range(start_step, numBatch):  # 总共的图片数目除以一个批数据10，所得的批数
            batch_input_ir = np.zeros((batch_size, patch_size_y, patch_size_x), dtype="float32")
            batch_input_vi = np.zeros((batch_size, patch_size_y, patch_size_x), dtype="float32")


            # batch_size = 70
            for patch_id in range(batch_size):
                # 图像在加载的时候已经转成了灰度图
                # 获取图像的高和宽
                # h, w, _= train_ir_data[image_id].shape
                h, w = train_ir_data[image_id].shape
                print(train_ir_data[image_id].shape)

                # 图像裁剪
                # 获取两个随机数x，y用于图像裁剪，取的是图像的左上角
                y = random.randint(0, h - patch_size_y - 1) # patch_size_y=224
                #  返回参数1与参数2之间的任一整数
                x = random.randint(0, w - patch_size_x - 1)

                batch_input_ir[patch_id, :, :] = train_ir_data[image_id][y: y + patch_size_y, x: x + patch_size_x]
                batch_input_vi[patch_id, :, :] = train_vi_data[image_id][y: y + patch_size_y, x: x + patch_size_x]
                image_id = (image_id + 1) % len(train_ir_data)

            # 保存图片
            for patch_id in range(batch_size):

                # print('=========================')
                # print(i)
                # print(patch_id)
                # print(batch_input_vi[patch_id])
                imsave(save_path+'vi/'+str(i)+'_'+str(patch_id)+'.bmp',batch_input_vi[patch_id].astype(np.uint8))
                imsave(save_path+'ir/'+str(i)+'_'+str(patch_id)+'.bmp',batch_input_ir[patch_id].astype(np.uint8))

print("[*] Finish training for phase %s." % train_phase)

