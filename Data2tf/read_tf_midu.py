#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/26|7:41 下午
# @Motto： Knowledge comes from decomposition

import tensorflow as tf

denseMap = {}
denseMap["d_st_2_did"] = 0
denseMap["d_st_7_copyrightid"] = 1
denseMap["d_st_2_app"] = 2
denseMap["d_st_7_firstcategoryid"] = 3
denseMap["d_st_7_secondcategoryid"] = 4
denseMap["d_st_7_thirdcategoryid"] = 5
denseMap["d_st_2_cpbookid"] = 6
denseMap["d_st_2_title"] = 7
denseMap["d_st_2_author"] = 8
denseMap["d_st_2_description"] = 9
denseMap["d_st_7_wordcount"] = 10
denseMap["d_st_3_endstatus"] = 11
denseMap["d_st_3_lengthstatus"] = 12
denseMap["d_st_3_status"] = 13
denseMap["d_st_3_salechapterno"] = 14
denseMap["d_st_3_saleprice"] = 15
denseMap["d_st_2_cptitle"] = 16
denseMap["d_st_3_chapterwords"] = 17
denseMap["d_st_7_readnum"] = 18
denseMap["d_st_7_likenum"] = 19
denseMap["d_st_7_hotnum"] = 20
denseMap["d_st_7_adjustnum"] = 21
denseMap["d_st_7_updatedat"] = 22
denseMap["d_dy_4_recvalidshow"] = 23
denseMap["d_dy_4_recread"] = 24
denseMap["d_dy_4_recreadlong"] = 25
denseMap["d_dy_4_recclick"] = 26
denseMap["d_dy_4_recshelf"] = 27
denseMap["d_dy_4_recshare"] = 28
denseMap["d_dy_3_recclickrate"] = 29
denseMap["d_dy_3_recavgreadlong"] = 30
denseMap["u_st_2_uid"] = 31
denseMap["u_st_2_sex"] = 32
denseMap["u_st_7_regtime"] = 33
denseMap["u_st_2_channel"] = 34
denseMap["u_st_2_phone_brand"] = 35
denseMap["u_st_2_city"] = 36
denseMap["c_2_hour"] = 37
denseMap["c_2_timedur"] = 38
denseMap["c_2_weekday"] = 39
denseMap["c_2_pageposition"] = 40
denseMap["c_2_pagedirect"] = 41
denseMap["c_2_pagedocpos"] = 42
denseMap["c_2_network"] = 43
denseMap["c_3_pagenum"] = 44

dense_list = [_[0] for _ in sorted(denseMap.items(), key=lambda _: _[1], reverse=False)]

sparseMap = {}
sparseMap["d_st_5_tag"] = 0  # multi 10
sparseMap["u_st_5_flavor"] = 1  # multi 34
sparseMap["u_dy_5_applist"] = 2  # multi 39
sparseMap["u_dy_5_readlist"] = 3  # multi 40
sparseMap["u_dy_6_readlist3"] = 4  # multi 41
sparseMap["u_dy_2_searchkeywordlist"] = 5  # multi 42
sparseMap["u_dy_5_addshelflist"] = 6  # multi 43
sparseMap["u_dy_5_sharelist"] = 7  # multi 44
sparseMap["u_dy_3_addshelflist"] = 8  # multi 45
sparseMap["u_dy_3_sharelist"] = 9  # multi 46
sparseMap["u_dy_6_favocate5"] = 10  # multi 47

spare_list = [_[0] for _ in sorted(sparseMap.items(), key=lambda _: _[1], reverse=False)]

feature_name = dense_list + spare_list


def get_tfrecords_example(sample_idx, feature_info, label_info, feature_value, label_value):
    tfrecords_features = {}

    tfrecords_features['sample_idx'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sample_idx))

    for f in feature_info:
        tfrecords_features[f] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=feature_value[f]))

    tfrecords_features[label_info] = tf.train.Feature(int64_list=tf.train.Int64List(value=label_value))

    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))


def read_tf_data(filename, write2file):
    writer = tf.python_io.TFRecordWriter(write2file)
    num = 0
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        num += 1
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        # Read data in specified format
        sample_idx = example.features.feature["sample_idx"].int64_list.value
        label = example.features.feature["label"].int64_list.value
        s1 = example.features.feature["dense"].int64_list.value
        s2 = example.features.feature["idx0"].int64_list.value
        s3 = example.features.feature["idx1"].int64_list.value
        s4 = example.features.feature["idx2"].int64_list.value
        s5 = example.features.feature["id_arr"].int64_list.value

        # print('--- dense ---')
        # for i in range(len(s1)):
        #     print(s1[i])
        # print('--- idx0, idx1, idx2, id_arr ---')
        # for i in range(len(s5)):
        #     print(s2[i], s3[i], s4[i], s5[i])
        # print('--- sample_idx, label ---')
        # print(sample_idx, label)
        # feature_value = {}

        label_value = [label[0]]
        print('--- transform: {} ---'.format(num))
        feature_value = {}
        for i in range(len(s1)):
            feature_value[dense_list[i]] = [s1[i]]

        for i in range(len(s5)):
            if spare_list[s3[i]] in feature_value.keys():
                feature_value[spare_list[s3[i]]].append(s5[i])
            else:
                feature_value[spare_list[s3[i]]] = [s5[i]]

        # for f, v in feature_value.items():
        #     print(f, v)
        example = get_tfrecords_example(sample_idx, dense_list + spare_list, 'label', feature_value, label_value)
        writer.write(example.SerializeToString())


if __name__ == '__main__':
    read_tf_data('../Data/midu/01_1580695206', '../Data/midu/raw_sample.tfrecord')
