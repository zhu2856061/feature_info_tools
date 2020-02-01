#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/26|3:20 下午
# @Motto： Knowledge comes from decomposition

import io
import yaml

def read_config(file_name):
    with io.open(file_name, encoding='utf-8') as f:
        feature_info = yaml.load(f, Loader=yaml.FullLoader)




        return feature_info
