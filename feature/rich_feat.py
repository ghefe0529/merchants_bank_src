# -*- coding: utf-8 -*-
# @Date    : 2018-06-22 16:49:14
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 丰富数据

import numpy as np
import pandas as pd
import math

def rich_baisc_math(X, Y, method):
    result = []
    for x,y in zip(X,Y):
        result.append(method(x,y))
    return result
# +
def my_add(x,y):
    return x+y
# -
def my_sub(x,y):
    return x-y
# *
def my_multi(x,y):
    return x*y
# /
def my_division(x,y):
    return x/y
# log(a,b)
def my_log_xy(x,y):
    return math.log(x,y)
# log(b,a)
def my_log_yx(x,y):
    return math.log(y,x)
# a^b
def my_exp_yx(x,y):
    return x**y
# b^a
def my_exp_xy(x,y):
    return y**x

# 两数平均
def my_avg(x,y):
    return (x+y)/2

# 标准差
def my_variance(X):
    return np.std(X, ddof=0)

# 平均数
def my_avg_list(X):
    return (sum(X)/len(X))

if __name__ == '__main__':
    X = [1,2,2]
    Y = [2,2,2]
    print(rich_baisc_math(X,Y,my_add))
    print(my_variance(X))