#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/25|8:18 下午
# @Motto： Knowledge comes from decomposition


import tensorflow as tf
from Tools.hasher import Hasher
from Tools.bucketer import Bucketer
from multiprocessing import Pool


def get_tfrecords_example(sample_idx, feature_info, label_info, feature_value, label_value):
    tfrecords_features = {}

    tfrecords_features['sample_idx'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample_idx))

    for f in feature_info:
        tfrecords_features[f['feature_name']] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=feature_value[f['feature_name']]))

    tfrecords_features[label_info] = tf.train.Feature(int64_list=tf.train.Int64List(value=label_value))

    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))


def multi_process_to_simple(i, feature_value_dict, feature_info, hasher, bucketer):
    sample_idx = [i]

    # 需要设置
    label_info = 'label'
    if feature_value_dict['label'] > 0:
        label_value = [1]
    else:
        label_value = [0]

    feature_value = {}
    for f in feature_info:
        if f['value_type'] == 2:
            feature_value[f['feature_name']] = [hasher.to_hash_64(
                str(f['feature_name']) + '=' + str(feature_value_dict[f['feature_name']]))]

        elif f['value_type'] == 5:
            multi_feature_value_lists = feature_value_dict[f['feature_name']].split('|')

            feature_value[f['feature_name']] = [hasher.to_hash_64(str(f['feature_name']) + '=' + str(m)) for m in
                                                multi_feature_value_lists]
        elif f['value_type'] == 3:
            tmp = bucketer[f['feature_name']].bucket_val(feature_value_dict[f['feature_name']])

            feature_value[f['feature_name']] = [hasher.to_hash_64(str(f['feature_name']) + '=' + str(tmp))]
        elif f['value_type'] == 4:
            tmp = bucketer[f['feature_name']].bucket_val(feature_value_dict[f['feature_name']])

            feature_value[f['feature_name']] = [hasher.to_hash_64(str(f['feature_name']) + '=' + str(tmp))]

    example = get_tfrecords_example(sample_idx, feature_info, label_info, feature_value, label_value)
    return example.SerializeToString()


def multi_process_to_feature_transform(filename, df_json, feature_info):
    print('*' * 18)
    print(feature_info)
    print('*' * 18)
    b = dict()
    for f in feature_info:
        if f['value_type'] == 3:
            b[f['feature_name']] = Bucketer(f['value_min'], f['value_max'], f['disperse_num'], 3)
        elif f['value_type'] == 4:
            b[f['feature_name']] = Bucketer(f['value_min'], f['value_max'], f['disperse_num'], 4)

    # 需要设置
    h = Hasher(2 ** 20)
    p = Pool(10)
    writer = tf.python_io.TFRecordWriter(filename)
    for i, feature_value_dict in enumerate(df_json):
        if i > 10000:
            break
        writer.write(p.apply_async(multi_process_to_simple, args=(i, feature_value_dict, feature_info, h, b)).get())
    p.close()
    p.join()
    writer.close()
