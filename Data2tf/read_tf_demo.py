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
        sample_idx = example.features.feature["sample_idx"].int64_list.value
        label = example.features.feature["label"].int64_list.value
        s1 = example.features.feature["u_st_2_uid"].int64_list.value
        s2 = example.features.feature["u_st_2_gender"].int64_list.value
        s3 = example.features.feature["u_st_3_age"].int64_list.value
        s4 = example.features.feature["u_st_2_zip-code"].int64_list.value
        s5 = example.features.feature["d_st_2_did"].int64_list.value
        s6 = example.features.feature["d_st_2_title"].int64_list.value
        s7 = example.features.feature["d_st_5_genres"].int64_list.value

        print(sample_idx, label, s1, s2, s3, s4, s5, s6, s7)
        break


if __name__ == '__main__':
    read_tf_data('../Data/raw_sample.tfrecord')


