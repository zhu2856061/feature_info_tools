#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/25|8:18 下午
# @Motto： Knowledge comes from decomposition


import pandas as pd
import tensorflow as tf
from hasher import Hasher
from multiprocessing import Pool

df_str = 'u_st_2_uid::d_st_2_did::label::Timestamp'.split('::')
user_df_str = 'u_st_2_uid::u_st_2_gender::u_st_3_age::u_st_2_occupation::u_st_2_zip-code'.split('::')
item_df_str = 'd_st_2_did::d_st_2_title::d_st_5_genres'.split('::')

df = pd.read_csv('../Data/movies_len/ratings.dat', sep='::', names=df_str, engine='python')
df.drop('Timestamp', axis=1, inplace=True)
user_df = pd.read_csv('../Data/movies_len/users.dat', sep='::', names=user_df_str, engine='python')
item_df = pd.read_csv('../Data/movies_len/movies.dat', sep='::', names=item_df_str, engine='python')


def get_tfrecords_example(sample_idx, label_arr, dense_id, idx0, idx1, idx2, sparse_id):
    tfrecords_features = {}

    tfrecords_features['sample_idx'] = \
        tf.train.Feature(int64_list=tf.train.Int64List(value=sample_idx))

    tfrecords_features['label'] = \
        tf.train.Feature(int64_list=tf.train.Int64List(value=label_arr))

    tfrecords_features['dense'] = \
        tf.train.Feature(int64_list=tf.train.Int64List(value=dense_id))

    tfrecords_features['idx0'] = \
        tf.train.Feature(int64_list=tf.train.Int64List(value=idx0))

    tfrecords_features['idx1'] = \
        tf.train.Feature(int64_list=tf.train.Int64List(value=idx1))

    tfrecords_features['idx2'] = \
        tf.train.Feature(int64_list=tf.train.Int64List(value=idx2))

    tfrecords_features['id_arr'] = \
        tf.train.Feature(int64_list=tf.train.Int64List(value=sparse_id))

    return tf.train.Example(
        features=tf.train.Features(feature=tfrecords_features))


def to_p(i, feature_value_dict, h):
    sample_idx = [i]
    dense_id = []
    idx0 = []
    idx1 = []
    idx2 = []
    id_arr = []

    if feature_value_dict['label'] > 3:
        label_arr = [1, 0]
    else:
        label_arr = [0, 1]

    for feature_name, feature_value in feature_value_dict.items():
        if feature_name == 'label':
            continue
        elif feature_name == 'd_st_5_genres':
            multi_feature_value_lists = feature_value.split('|')
            for ind, multi_feature_value in enumerate(multi_feature_value_lists):
                idx0.append(i)
                idx1.append(ind)
                idx2.append(ind)
                id_arr.append(h.to_hash_64(str(feature_name) + '=' + str(multi_feature_value)))
        else:
            dense_id.append(h.to_hash_64(str(feature_name) + '=' + str(feature_value)))

    example = get_tfrecords_example(sample_idx, label_arr, dense_id, idx0, idx1, idx2, id_arr)
    return example.SerializeToString()


def to_data(filename):
    h = Hasher(2 ** 20)
    df_all = pd.merge(pd.merge(df, user_df, on='u_st_2_uid', how='left'), item_df, on='d_st_2_did', how='left')
    print(df_all.dtypes)
    print('*' * 18)
    print(df_all.count())
    print('*' * 18)
    df_json = df_all.to_dict('records')

    # multiprocessing
    p = Pool(10)
    results = []

    for i, feature_value_dict in enumerate(df_json):
        # if i > 100:
        #     break
        results.append(p.apply_async(to_p, args=(i, feature_value_dict, h)))
    print('*' * 18)
    print(len(results))
    print('*' * 18)
    p.close()
    p.join()
    writer = tf.python_io.TFRecordWriter(filename)
    for r in results:
        writer.write(r.get())
    writer.close()


if __name__ == '__main__':
    filename = '../Data/sample.tfrecord'
    to_data(filename)
