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
from ColAucDiff.input_fn import ReadData, FormatData
from ColAucDiff.fm_model import FM
from Tools.config import read_config
from Tools.auc import AUCUtil
from tqdm import tqdm
from Tools.logger import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(2020)
np.random.seed(2020)
tf.set_random_seed(2020)

# --- step02: params data ---
batch_size = 8
latent = 33
learning_rate = 0.001
l2 = 0.8
l1 = 0.9
epochs = 15
best_auc = 0.0


def _eval(sess, model, test_len, fd, auc_op):
    global best_auc
    batch_per_epoch = (test_len + batch_size - 1) // batch_size
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
def train_epochs(fd, idx_len, train_len, test_len):
    global best_auc
    print('*' * 16)
    print(train_len, test_len)
    print('*' * 16)

    auc_op = AUCUtil()
    model = FM(latent, learning_rate, l2, l1, idx_len)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start_time = time.time()
        for i in range(epochs):
            loss_sum = 0.0
            batch_per_epoch = (train_len + batch_size - 1) // batch_size
            for _ in tqdm(range(batch_per_epoch)):
                xs, ys, st = fd.next_train_batch(batch_size)

                loss = model.train(sess, xs, st, ys)
                loss_sum += loss

                if model.global_step.eval() % 1000 == 0:
                    auc = _eval(sess, model, test_len, fd, auc_op)
                    logger.get_log().info('Epoch %d Global_step %d\tTrain_loss: %.5f\tTest_loss: %.5f\tEval_AUC: %.5f' %
                                          (i, model.global_step.eval(), loss_sum / 1000, auc['loss'], auc['auc']))
                    sys.stdout.flush()
                    loss_sum = 0.0

            auc = _eval(sess, model, test_len, fd, auc_op)
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
    sample = '../Data/midu/raw_sample.tfrecord'
    config = '../Config/feature_midu.yaml'
    feature_info = read_config(config)
    res_auc = {}
    res_aucdiff = {}

    # read data
    rd = ReadData(sample, feature_info, 2020)
    idx = rd.get_idx()

    # basic auc
    train, test = rd.rand_col()
    fd_tmp = FormatData(train, test, idx)
    res_auc['basic'] = train_epochs(fd_tmp, len(idx), len(train), len(test))
    res_aucdiff['basic'] = 0

    for f in feature_info:
        train_data, test_data = rd.rand_col(f['feature_name'])
        fd_tmp = FormatData(train_data, test_data, idx)

        res_auc[f['feature_name']] = train_epochs(fd_tmp, len(idx), len(train), len(test))

        res_aucdiff[f['feature_name']] = res_auc[f['feature_name']] - res_auc['basic']

    import pandas as pd
    df = pd.DataFrame([res_auc, res_aucdiff], index=['auc', 'auc_diff'])
    print(df)
    df.to_csv('../Result/ans02.csv')
