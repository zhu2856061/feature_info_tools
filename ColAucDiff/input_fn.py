#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/27|5:15 下午
# @Motto： Knowledge comes from decomposition
#
from Tools.util import shuffle_in_unison_scary
import tensorflow as tf
from collections import defaultdict
import random


def input_fn(sample_path, config_path, rand_col=None):
    from Tools.config import read_config
    rd = ReadData(sample_path, read_config(config_path), 2020)
    train, test = rd.rand_col(rand_col)
    idx = rd.get_idx()
    fd = FormatData(train, test, idx)
    return fd, idx, len(train), len(test)


class ReadData:
    def __init__(self, filename, feature_info, seed):
        self.filename = filename
        self.feature_info = feature_info
        self.f_dic = {}
        self.unique = defaultdict(set)
        self.train = []
        self.test = []
        random.seed(seed)
        for serialized_example in tf.python_io.tf_record_iterator(self.filename):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            sdic = {}
            for f in self.feature_info:
                tmp = example.features.feature[f['feature_name']].int64_list.value
                self.unique[f['feature_name']].update(set(tmp))
                for t in tmp:
                    if self.f_dic.get(t, None) is None:
                        self.f_dic[t] = len(self.f_dic)

                sdic[f['feature_name']] = tmp

            tmp = example.features.feature['label'].int64_list.value
            sdic['label'] = tmp
            # 需要设置，划分train test
            if random.randint(0, 9) >= 7:
                self.test.append(sdic)
            else:
                self.train.append(sdic)

    def get_idx(self):
        return self.f_dic

    def rand_col(self, col_name=None):
        train = []
        test = []
        if col_name is None:
            for tmp in self.train:
                line = []
                for k, v in tmp.items():
                    if k == 'label':
                        continue
                    for t in v:
                        line.append(t)
                train.append([line, tmp['label']])

            for tmp in self.test:
                line = []
                for k, v in tmp.items():
                    if k == 'label':
                        continue
                    for t in v:
                        line.append(t)
                test.append([line, tmp['label']])

        else:
            for tmp in self.train:
                line = []
                for k, v in tmp.items():
                    if k == 'label':
                        continue
                    if k == col_name:
                        v = random.sample(self.unique[k], len(v))
                    for t in v:
                        line.append(t)

                train.append([line, tmp['label']])

            for tmp in self.test:
                line = []
                for k, v in tmp.items():
                    if k == 'label':
                        continue
                    if k == col_name:
                        v = random.sample(self.unique[k], len(v))
                    for t in v:
                        line.append(t)

                test.append([line, tmp['label']])

        return train, test


class FormatData:
    def __init__(self, train, test, f_dic):

        self.train_idx = []
        self.train_y = []
        self.train_size = len(train)
        self._train_current = 0
        self.train_index = list(range(self.train_size))

        self.test_idx = []
        self.test_y = []
        self.test_size = len(test)
        self._test_current = 0
        self.test_index = list(range(self.test_size))

        for t in train:
            self.train_idx.append([f_dic.get(i, 0) for i in t[0]])
            self.train_y.append(t[1])
        for t in test:
            self.test_idx.append([f_dic.get(i, 0) for i in t[0]])
            self.test_y.append(t[1])

    def next_train_batch(self, batch_size):

        if self._train_current >= self.train_size:
            self._train_current = 0
        if self._train_current == 0:
            shuffle_in_unison_scary(self.train_idx, self.train_y)

        start = self._train_current
        end = min(self._train_current + batch_size, self.train_size)
        self._train_current = end

        xs = self.train_idx[start:end]
        st = [len(_) for _ in xs]
        max_len = max(st)
        xs = [_ + [0] * (max_len - len(_)) for _ in xs]
        ys = self.train_y[start:end]
        return xs, ys, st

    def next_test_batch(self, batch_size):

        if self._test_current >= self.test_size:
            self._test_current = 0
        if self._test_current == 0:
            shuffle_in_unison_scary(self.test_idx, self.test_y)

        start = self._test_current
        end = min(self._test_current + batch_size, self.test_size)
        self._test_current = end

        xs = self.test_idx[start:end]
        st = [len(_) for _ in xs]
        max_len = max(st)
        xs = [_ + [0] * (max_len - len(_)) for _ in xs]
        ys = self.test_y[start:end]
        return xs, ys, st

# if __name__ == '__main__':
#     from Tools.config import read_config
#     from ColAucDiff.rand_col_value import ReadData
#
#     rd = ReadData('../Data/raw_sample.tfrecord', read_config('../Config/feature_test.yaml'))
#     traintmp, testtmp = rd.rand_col()
#     train_r, test_r = rd.rand_col('d_st_5_genres')
#     idx = rd.get_idx()
#
#     fd = FormatData(traintmp, testtmp, idx)
#
#     xstmp, ystmp, sttmp = fd.next_train_batch(128)
#     print('* ' * 16)
#     for x in xstmp:
#         print(x)
#     print(ystmp)
#     print('* ' * 16)
