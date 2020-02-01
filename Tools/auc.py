#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/26|9:53 下午
# @Motto： Knowledge comes from decomposition

import numpy as np
from sklearn.metrics import roc_auc_score


class AUCUtil(object):
    def __init__(self):
        self.reset()

    def add(self, loss, g=np.array([]), p=np.array([])):
        self.loss.append(loss)
        self.ground_truth += g.flatten().tolist()
        self.prediction += p.flatten().tolist()

    def calc(self):
        return {
            "loss_num": len(self.loss),
            "loss": np.array(self.loss).mean(),
            "auc_num": len(self.ground_truth),
            "auc": roc_auc_score(self.ground_truth, self.prediction) if len(self.ground_truth) > 0 else 0  # noqa
        }

    def calc_str(self):
        res = self.calc()
        return "loss: %f(%d), auc: %f(%d)" % (
            res["loss"], res["loss_num"],
            res["auc"], res["auc_num"]
        )

    def reset(self):
        self.loss = []
        self.ground_truth = []
        self.prediction = []


class GAUCUtil(object):
    def __init__(self):
        self.reset()

    def add(self, loss, g=np.array([]), p=np.array([]), sid=np.array([])):
        self.loss.append(loss)
        self.ground_truth += g.flatten().tolist()
        self.prediction += p.flatten().tolist()

        tmp_store = []
        tmp_g = g.flatten()
        tmp_p = p.flatten()
        tmp_s = sid.flatten()
        for i in range(sid.shape[0]):
            tmp_store.append([tmp_s[i], tmp_g[i], tmp_p[i]])
        self.store += tmp_store

    def calc(self):
        self.store = sorted(self.store, key=lambda d: d[0])
        last_index = 0
        g_arr = []
        p_arr = []
        gauc_sum = 0.
        valid_pv_sum = 0
        valid_uv_sum = 0

        for idx in range(len(self.store)):
            if self.store[idx][0] != self.store[last_index][0]:
                counter = np.array(g_arr).astype(np.int32).sum()
                if not (counter == 0 or counter == len(g_arr)):
                    gauc_sum += roc_auc_score(g_arr, p_arr) * len(g_arr)
                    valid_pv_sum += len(g_arr)
                    valid_uv_sum += 1

                g_arr = []
                p_arr = []
                last_index = idx

            g_arr.append(self.store[idx][1])
            p_arr.append(self.store[idx][2])

        return {
            "loss_num": len(self.loss),
            "loss": np.array(self.loss).mean(),
            "auc_num": len(self.ground_truth),
            "auc": roc_auc_score(self.ground_truth, self.prediction) if len(self.ground_truth) > 0 else 0,  # noqa
            "gauc_num": valid_pv_sum,
            "gauc_uv_num": valid_uv_sum,
            "gauc": gauc_sum / valid_pv_sum if valid_pv_sum > 0 else 0
        }

    def calc_str(self):
        res = self.calc()
        return "loss: %f(%d), auc: %f(%d), gauc: %f(%d, %d)" % (
            res["loss"], res["loss_num"],
            res["auc"], res["auc_num"],
            res["gauc"], res["gauc_num"],
            res["gauc_uv_num"]
        )

    def reset(self):
        self.loss = []
        self.ground_truth = []
        self.prediction = []
        self.store = []
