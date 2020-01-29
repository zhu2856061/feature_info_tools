#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/29|2:05 下午
# @Motto： Knowledge comes from decomposition

from Data2tf.data2tf import multi_process_to_feature_transform
import pandas as pd
from Tools.config import read_config

df_str = 'u_st_2_uid::d_st_2_did::label::Timestamp'.split('::')
user_df_str = 'u_st_2_uid::u_st_2_gender::u_st_3_age::u_st_2_occupation::u_st_2_zip-code'.split('::')
item_df_str = 'd_st_2_did::d_st_2_title::d_st_5_genres'.split('::')

df = pd.read_csv('ratings.dat', sep='::', names=df_str, engine='python')
df.drop('Timestamp', axis=1, inplace=True)
user_df = pd.read_csv('users.dat', sep='::', names=user_df_str, engine='python')
item_df = pd.read_csv('movies.dat', sep='::', names=item_df_str, engine='python')

df_all = pd.merge(pd.merge(df, user_df, on='u_st_2_uid', how='left'), item_df, on='d_st_2_did', how='left')
print('*' * 18)
print(len(df_all))
print('*' * 18)

filename = 'raw_sample.tfrecord'
df_json = df_all.to_dict('records')
feature_info = read_config('../../Config/feature_movies_len.yaml')

multi_process_to_feature_transform(filename, df_json, feature_info)
