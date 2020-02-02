#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/17|2:39 下午
# @Motto： Knowledge comes from decomposition

import pandas as pd
import tensorflow as tf
from collections import defaultdict
from Tools.config import read_config


# 覆盖度	平均长度	unique个数	unique占比
def col_info(filename, feature_info, result_info):
    num = 0
    coverage_tmp = defaultdict(int)
    average_tmp = defaultdict(int)
    unique_tmp = defaultdict(set)
    unique_count_tmp = defaultdict(defaultdict)
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        # Read data in specified format
        num += 1
        for f in feature_info:
            tmp = example.features.feature[f['feature_name']].int64_list.value
            coverage_tmp[f['feature_name']] += 1 if len(tmp) >= 1 else 0
            average_tmp[f['feature_name']] += len(tmp)
            unique_tmp[f['feature_name']].update(set(tmp))
            if f['value_type'] == 2:
                unique_count_tmp[f['feature_name']][tmp[0]] += 1

    unique_count = dict()
    for k, v in unique_count_tmp.items():
        unique_count[k] = sorted(v.items(), key=lambda _: _[1], reverse=False)

    coverage = dict()
    for k, v in coverage_tmp.items():
        coverage[k] = float(v) / num

    average = dict()
    for k, v in average_tmp.items():
        average[k] = float(v) / num

    unique = dict()
    unique_rate = dict()
    for k, v in unique_tmp.items():
        unique[k] = len(v)
        unique_rate[k] = float(len(v)) / num

    df = pd.DataFrame([coverage, average, unique, unique_rate], index=['coverage', 'average', 'unique', 'unique_rate'])
    print(df)
    df.to_csv(result_info)


if __name__ == '__main__':
    # col_info('../Data/movies_len/raw_sample.tfrecord',
    #          read_config('../Config/feature_movies_len.yaml'), '../Result/ans01.csv')

    col_info('../Data/adult_uci/raw_adult_sample.tfrecord',
             read_config('../Config/feature_adult.yaml'), '../Result/ans01.csv')

    # print('coverage')
    # print(coverage)
    # print('*'*16)
    # print('average')
    # print(average)
    # print('*' * 16)
    # print('unique')
    # print(unique)
    # print('*' * 16)
    # print('unique_rate')
    # print(unique_rate)
