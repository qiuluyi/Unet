import re
import os, sys
import numpy as np
import pdb
from PIL import Image
import h5py
import argparse
import pandas as pd
import xlwt
import xlrd
from xlutils.copy import copy
import keras.backend as K
import tensorflow as tf
from data import *

from unet import unet

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs [1000]')
parser.add_argument('--lr', type=float, default=2e-4, help='learning_rate [2e-4]')
parser.add_argument('--batch_size', type=int, default=2,
                    help='size of mini-batch [2]')
parser.add_argument('--img_rows', type=int, default=256,
                    help='image rows [256]')
parser.add_argument('--img_cols', type=int, default=256,
                    help='image cols [512]')
# loss初始值，可调，如果代码跑完之后没保存模型可将该值调大一点
parser.add_argument('--r_che', type=float, default=0.25,
                    help='Reading checkpoints [False]')
parser.add_argument('--keep_prob', type=float, default=0.5,
                    help='keep_prob [0.8]')
parser.add_argument('--re_logdir', type=str, default="model/checkpoint",
                    help='re_logdir')
parser.add_argument('--wr_logdir', type=str, default="model/result.ckpt",
                    help='wr_logdir')
parser.add_argument('--fir', type=bool, default=False,
                    help='first read unet')
parser.add_argument('--excel_path', type=str, default="unet_result.xls",
                    help='excel_path')

args = parser.parse_args()

image_size = [args.img_rows, args.img_cols]


class UNET():
    def __init__(self, is_fir=args.fir,
                 img_rows=args.img_rows, img_cols=args.img_cols,
                 lr=args.lr, batch_size=args.batch_size
                 ):

        self.keep_prob = tf.placeholder(tf.float32)
        self.unet = unet(keep_prob=self.keep_prob)
        self.unet.summary()
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.x = tf.placeholder(tf.float32,
                                (None, self.img_rows, self.img_cols, 1),
                                name='x')
        self.y = tf.placeholder(tf.float32,
                                (None, self.img_rows, self.img_cols, 1),
                                name='y')

        self.lr = tf.Variable(2e-4, trainable=False, name='learning_rate')
        self.y_ = self.unet(self.x)
        self.unet_loss = self.get_unet_cost(self.y_, self.y, batch_size=batch_size)
        self.u_opt = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.unet_loss)
        self.saver = tf.train.Saver()
        self.jacard_dis, self.jacard_coef = self.get_jac_dis_cost(self.convert_img(self.y_), self.y, batch_size)
        self.dice_coef = self.dice_coef(self.convert_img(self.y_), self.y)
        self.sess = tf.Session()
        K.set_session(self.sess)

    def get_unet_cost(self, logits, labels, batch_size):
        flat_logits = tf.reshape(logits, [batch_size, -1])
        flat_labels = tf.reshape(labels, [batch_size, -1])

        loss = tf.reduce_mean(
            -flat_labels * tf.log(flat_logits + 1e-6) - (1 - flat_labels) * tf.log(
                1 - flat_logits + 1e-6))
        return loss

    def load_data(self):

        if os.path.exists("npy_data/train_images.npy") and os.path.exists("npy_data/train_masks.npy"):
            pass
        else:
            make_data.make_train_npy()
        if os.path.exists("npy_data/test_images.npy") and os.path.exists("npy_data/test_masks.npy"):
            pass
        else:
            make_data.make_test_npy()

        imgs_train, mask_train = load_train_data()
        imgs_val, mask_val = load_val_data()

        return imgs_train, mask_train, imgs_val, mask_val

    # jaccard评价函数
    def get_jac_dis_cost(self, logits, labels, batch_size):
        flat_logits = tf.reshape(logits, [batch_size, -1])
        flat_labels = tf.reshape(labels, [batch_size, -1])
        tp = tf.reduce_sum(tf.multiply(flat_labels, flat_logits), 1)
        t = tf.reduce_sum(tf.multiply(flat_labels, flat_labels), 1)
        p = tf.reduce_sum(tf.multiply(flat_logits, flat_logits), 1)
        l = 1 - (tp / (t + p - tp + 1e-6))
        jac_coef = tp / (t + p - tp + 1e-6)
        return tf.reduce_mean(l), jac_coef

    def convert_img(self, img):
        fill_zero = tf.zeros_like(img,dtype=tf.float32)
        fill_ones = tf.ones_like(img,dtype=tf.float32)
        convert_img = tf.where(tf.less(img, 0.5), fill_zero, fill_ones)
        return convert_img

    # dice评价函数
    def dice_coef(self, y_true, y_pred):
        smooth = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def train(self, is_r_ch=args.r_che, batch_size=args.batch_size, epochs=args.epochs, re_modeldir=args.re_logdir,
              wr_modeldir=args.wr_logdir, excel_save_path=args.excel_path):

        print("loading data")
        imgs_train, mask_train, imgs_val, mask_val = self.load_data()

        assert imgs_train.shape[0] == mask_train.shape[0]
        num_tra = imgs_train.shape[0]

        assert imgs_val.shape[0] == mask_val.shape[0]
        num_val = imgs_val.shape[0]

        print('the number of train data is {}'.format(num_tra))
        print('the number of val data is {}'.format(num_val))

        print("loading data done")

        num_batches = int(num_tra / batch_size)

        self.sess.run(tf.global_variables_initializer())

        global_step = 0

        if os.path.exists(re_modeldir):
            print("loading checkpoint")
            self.saver.restore(self.sess, wr_modeldir)
            print("load down")
        excel_i = 0
        book = xlwt.Workbook()
        sheet = book.add_sheet('test1')  # 创建sheet,并命名
        list_value = ["idx", "jarccard", "dice", "test_loss"]
        for l in range(len(list_value)):
            sheet.write(excel_i, l, list_value[l])
        book.save(excel_save_path)  # 一般文件打不开可能是扩展名的问题

        for e in range(global_step, epochs):

            permutation = list(np.random.permutation(num_tra))

            imgs_training = imgs_train[permutation]
            mask_training = mask_train[permutation]

            for idx in range(num_batches):

                imgs = imgs_training[idx * batch_size:(idx + 1) * batch_size]
                imgs_lab = mask_training[idx * batch_size:(idx + 1) * batch_size]
                self.sess.run(self.u_opt, feed_dict={self.x: imgs, self.y: imgs_lab, self.keep_prob: 0.5})

                if e % 1 == 0:
                    excel_list = []
                    if idx % 10 == 0:
                        print('train_loss')
                        [loss_value] = self.sess.run([self.unet_loss],
                                                     feed_dict={self.x: imgs, self.y: imgs_lab, self.keep_prob: 0.5})
                        print("e %d idx %d [train_loss: %f]" % (e, idx, loss_value))

# ##################################################################################
                        # 每十次向excel中写入一次数据，即迭代的loss和jaccaed、dice评价系数
                        loss_value_test = 0
                        for j in range(3):
                            [loss_value] = self.sess.run([self.unet_loss],
                                                         feed_dict={
                                                             self.x: imgs_val[j * batch_size:(j + 1) * batch_size],
                                                             self.y: mask_val[j * batch_size:(j + 1) * batch_size],
                                                             self.keep_prob: 1.0})

                            loss_value_test += loss_value
                        loss_value_test_1 = 0
                        jacard = 0
                        dice = 0

                        for j in range(3):
                            [loss_value_1, dice_coef, jacard_coef] = self.sess.run(
                                [self.unet_loss, self.dice_coef, self.jacard_coef],
                                feed_dict={self.x: imgs_val[j * batch_size:(j + 1) * batch_size],
                                           self.y: mask_val[j * batch_size:(j + 1) * batch_size],
                                           self.keep_prob: 1.0})
                            loss_value_test_1 += loss_value_1
                            dice += dice_coef
                            jacard += jacard_coef

                        excel_i += 1
                        excel_list.append(excel_i)
                        excel_list.append((jacard[0] + jacard[1]) / 6)
                        excel_list.append(dice / 3)
                        excel_list.append(loss_value_test_1 / 3)
                        workbook = xlrd.open_workbook(excel_save_path)
                        new_book = copy(workbook)
                        newsheet = new_book.get_sheet(0)
                        for j in range(len(excel_list)):
                            newsheet.write(excel_i, j, excel_list[j])
                        new_book.save(excel_save_path)
# ##################################################################################
                        print('test_loss')

                        print("e %d idx %d [test_loss: %f]" % (e, idx, loss_value_test / 3))
                        x = (loss_value_test / 3)

                        if x < is_r_ch:
                            for j in range(3):                # 保存图片，测试图片几张即为几
                                imgs_test_picture = self.sess.run(self.y_, feed_dict={self.x: imgs_val[j * batch_size:(j + 1) * batch_size],
                                               self.y: mask_val[j * batch_size:(j + 1) * batch_size],
                                               self.keep_prob: 1.0})
                                for i in range(batch_size):
                                    img = imgs_test_picture[i]
                                    img[img > 0.5] = 1          # 图片二值化
                                    imgs_input = imgs_val[j * batch_size:(j + 1) * batch_size][i]
                                    imgs_label = mask_val[j * batch_size:(j + 1) * batch_size][i]
                                    img = array_to_img(img)
                                    imgs_input = array_to_img(imgs_input)
                                    imgs_label = array_to_img(imgs_label)

                                    if os.path.exists('./result/unet_output'):
                                        pass
                                    else:
                                        os.makedirs('./result/unet_output')
                                    if os.path.exists('./result/input'):
                                        pass
                                    else:
                                        os.makedirs('./result/input')
                                    if os.path.exists('./result/label'):
                                        pass
                                    else:
                                        os.makedirs('./result/label')
                                    img.save("result/unet_output/j is %d and i is %d.jpg" % (j, i))
                                    imgs_input.save("result/input/j is %d and i is %d.jpg" % (j, i))
                                    imgs_label.save("result/label/j is %d and i is %d.jpg" % (j, i))
                            print("测试图片存储成功")
                            is_r_ch = x
                            if os.path.exists("model"):
                                pass
                            else:
                                os.makedirs("model")
                            self.saver.save(self.sess, wr_modeldir)  # 保存模型


if __name__ == '__main__':
    myunet = UNET()
    myunet.train()
