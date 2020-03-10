#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/26|7:41 下午
# @Motto： Knowledge comes from decomposition

import tensorflow as tf


def read_tf_data(filename):
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        # Read data in specified format
        # sample_idx = example.features.feature["sample_idx"].int64_list.value
        # for k, v in example.features.feature.items():
        #     print(k)
        print(example)
        break


if __name__ == '__main__':
    read_tf_data('../Data/test/part-1582221558157461509')


