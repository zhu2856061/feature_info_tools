#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/30|10:56 下午
# @Motto： Knowledge comes from decomposition
import logging


class Logger:
    def __init__(self, logger_name):
        # 创建一个logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        log_name = "../Logs/{}.log".format(logger_name)
        fh = logging.FileHandler(log_name, encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # 创建一个handler，用于将日志输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_log(self):
        """定义一个函数，回调logger实例"""
        return self.logger


# if __name__ == '__main__':
#     t = Logger("hmk").get_log().debug("User %s is loging" % 'jeck')
