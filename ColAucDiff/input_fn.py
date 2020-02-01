#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/27|5:15 下午
# @Motto： Knowledge comes from decomposition
#
from Tools.util import shuffle_in_unison_scary


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


def input_fn():
    from Tools.config import read_config
    from ColAucDiff.rand_col_value import ReadData
    rd = ReadData('../Data/movies_len/raw_sample.tfrecord', read_config('../Config/feature_movies_len.yaml'))
    train, test = rd.rand_col()
    # train_r, test_r = rd.rand_col('d_st_5_genres')
    idx = rd.get_idx()

    fd = FormatData(train, test, idx)
    return fd


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
