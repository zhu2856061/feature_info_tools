#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/27|5:42 下午
# @Motto： Knowledge comes from decomposition

import numpy as np


def shuffle_in_unison_scary(X, y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
