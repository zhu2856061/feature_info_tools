#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/2/2|11:53 上午
# @Motto： Knowledge comes from decomposition

import pandas as pd
import tensorflow as tf
from collections import defaultdict
from Tools.config import read_config


def col_info(filename, feature_info, result_info):
    num = 0
    unique_tmp = defaultdict(defaultdict)
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        # Read data in specified format
        num += 1
        for f in feature_info:
            if f['value_type'] == 2:
                tmp = example.features.feature[f['feature_name']].int64_list.value
                unique_tmp[f['feature_name']][tmp[0]] += 1

    unique_count = dict()
    for k, v in unique_tmp.items():
        unique_count[k] = sorted(v.items(), key=lambda _: _[1], reverse=False)

    return unique_count


if __name__ == '__main__':
    # col_info('../Data/movies_len/raw_sample.tfrecord',
    #          read_config('../Config/feature_movies_len.yaml'), '../Result/ans01.csv')

    col_info('../Data/adult_uci/raw_adult_sample.tfrecord',
             read_config('../Config/feature_adult.yaml'), '../Result/ans01.csv')
