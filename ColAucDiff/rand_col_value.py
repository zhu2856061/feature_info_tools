#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/26|10:01 下午
# @Motto： Knowledge comes from decomposition

import tensorflow as tf
from collections import defaultdict

import random

'''
readData
1. 
'''


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

# if __name__ == '__main__':
#     from Tools.config import read_config
#     rd = ReadData('../Data/raw_sample.tfrecord', read_config('../Config/feature_test.yaml'))
#     train, test = rd.rand_col()
#     train_r, test_r = rd.rand_col('d_st_5_genres')
#     idx = rd.get_idx()
#     print(len(idx))
#
#     for t in train_r:
#         print(t)
#         break
#     for t in train:
#         print(t)
#         break
