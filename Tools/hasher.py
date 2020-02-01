#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/25|10:54 下午
# @Motto： Knowledge comes from decomposition
from sklearn.feature_extraction import FeatureHasher
import numpy as np


class Hasher:
    def __init__(self, n):
        self.h = FeatureHasher(n_features=n, input_type='string', dtype=np.int64, alternate_sign=False)

    def to_hash_64(self, s):
        cur_idx = self.h.transform([[s]]).indices[0]
        int64_max = np.iinfo(np.int64).max
        ind = -int64_max + cur_idx
        return ind

    def to_hash(self, s):
        cur_idx = self.h.transform([[s]]).indices[0]
        return cur_idx


if __name__ == '__main__':
    h = Hasher(10)
    ind = h.to_hash('az')
    print(ind)
