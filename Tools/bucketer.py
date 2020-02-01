#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/26|12:32 下午
# @Motto： Knowledge comes from decomposition

import math


class Bucketer:

    def __init__(self, min_value, max_value, num, dtype):
        self.dtype = dtype
        if dtype == 4:
            max_value = math.log(max_value) if max_value > 0 else max_value
            min_value = math.log(min_value) if min_value > 0 else min_value

            st = float(max_value - min_value) / num
            self.dic = [(float("-inf"), min_value, 0), (max_value, float("inf"), num + 1)]
            for i in range(num):
                self.dic.append((min_value + i * st, min_value + (i + 1) * st, i + 1))

        else:
            st = float(max_value - min_value) / num
            self.dic = [(float("-inf"), min_value, 0), (max_value, float("inf"), num + 1)]
            for i in range(num):
                self.dic.append((min_value + i * st, min_value + (i + 1) * st, i + 1))

    def bucket_val(self, x):
        try:
            if self.dtype == 4:
                x = math.log(x) if x > 0 else x
                for r in self.dic:
                    if r[0] <= x < r[1]:
                        return r[2]
            else:
                for r in self.dic:
                    if r[0] <= x < r[1]:
                        return r[2]
        except:
            return 0
