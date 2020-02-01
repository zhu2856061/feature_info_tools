#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/20|10:49 下午
# @Motto： Knowledge comes from decomposition


from multiprocessing import Pool


def test(p):
    return p


if __name__ == "__main__":
    pool = Pool(processes=10)
    result = []
    for i in range(50000):
        '''
        for循环执行流程：
        （1）添加子进程到pool，并将这个对象（子进程）添加到result这个列表中。（此时子进程并没有运行）
        （2）执行子进程（同时执行10个）
        '''
        result.append(pool.apply_async(test, args=(i,)))  # 维持执行的进程总数为10，当一个进程执行完后添加新进程.
    pool.close()
    pool.join()
    '''
    遍历result列表，取出子进程对象，访问get()方法，获取返回值。（此时所有子进程已执行完毕）
    '''
    for i in result:
        print(i.get())
