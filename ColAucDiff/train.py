#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/28|8:53 下午
# @Motto： Knowledge comes from decomposition
import os
import time
import random
import numpy as np
import tensorflow as tf
import sys
from ColAucDiff.input_fn import FormatData
from ColAucDiff.model import FM
from Tools.config import read_config
from ColAucDiff.rand_col_value import ReadData
from Tools.auc import AUCUtil
from tqdm import tqdm
from Tools.logger import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(2020)
np.random.seed(2020)
tf.set_random_seed(2020)

# --- step01: read data ---


# --- step02: params data ---
batch_size = 8
latent = 33
learning_rate = 0.0005
l2 = 0.8
l1 = 0.9
epochs = 15
best_auc = 0.0


def _eval(sess, model, test, fd, auc_op):
    global best_auc
    batch_per_epoch = (len(test) + batch_size - 1) // batch_size
    for _ in range(batch_per_epoch):
        xs, ys, st = fd.next_test_batch(batch_size)
        arr = model.test(sess, xs, st, ys)

        auc_op.add(arr[0], np.array(ys), arr[1])

    auc_res = auc_op.calc()
    if best_auc < auc_res['auc']:
        best_auc = auc_res['auc']
        model.save(sess, 'save_path/ckpt')
    return auc_res


# --- step03: train data ---
def train_epochs(train, test):
    global best_auc
    print('*' * 16)
    print(len(train), len(test))
    print('*' * 16)
    idx = rd.get_idx()
    fd = FormatData(train, test, idx)
    auc_op = AUCUtil()
    feature_hasher_size = len(idx)
    model = FM(latent, learning_rate, l2, l1, feature_hasher_size)

    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start_time = time.time()
        for i in range(epochs):
            loss_sum = 0.0
            batch_per_epoch = (len(train) + batch_size - 1) // batch_size
            for _ in tqdm(range(batch_per_epoch)):
                xs, ys, st = fd.next_train_batch(batch_size)

                loss = model.train(sess, xs, st, ys)
                loss_sum += loss

                if model.global_step.eval() % 1000 == 0:
                    auc = _eval(sess, model, test, fd, auc_op)
                    logger.get_log().info('Epoch %d Global_step %d\tTrain_loss: %.5f\tTest_loss: %.5f\tEval_AUC: %.5f' %
                                          (i, model.global_step.eval(), loss_sum / 1000, auc['loss'], auc['auc']))
                    sys.stdout.flush()
                    loss_sum = 0.0

            auc = _eval(sess, model, test, fd, auc_op)
            logger.get_log().info('Epoch %d DONE\tCost time: %.2f Train_loss: %.5f\tTest_loss: %.5f\tEval_AUC: %.5f' % (
                i, time.time() - start_time, loss_sum / batch_per_epoch, auc['loss'], auc['auc']))

            start_time = time.time()
            sys.stdout.flush()

    print('*' * 16)
    print('best test_auc:', best_auc)
    res = best_auc
    best_auc = 0.0
    print('*' * 16)
    sys.stdout.flush()
    return res


if __name__ == '__main__':
    logger = Logger('fm_auc_diff')
    feature_info = read_config('../Config/feature_adult.yaml')
    rd = ReadData('../Data/adult_uci/raw_adult_sample.tfrecord', feature_info, 2020)

    train_data, test_data = rd.rand_col()
    auc_value = train_epochs(train_data, test_data)
    print('auc: ', auc_value)
