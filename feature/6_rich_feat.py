# -*- coding: utf-8 -*-
# @Date    : 2018-06-22 16:49:14
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 丰富数据

import numpy as np
import pandas as pd
import math


# common_path = r'~/Documents/Study/Python/merchants_bank/'
common_path = r'~/Documents/merchants_bank'
# train 
train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'

train_new_agg_path = common_path + r'/data/feature/6_train_new_agg.csv'

# test
test_agg_path = common_path + r'/data/corpus/output/test_agg.csv'

test_new_agg_path = common_path + r'/data/feature/6_test_new_agg.csv'


def rich_baisc_math(X, Y, method):
    # print(type(method))
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
    if y == 0:
        return 0
    return x/y
# log(a,b)
def my_log_xy(x,y):
    return math.log(x,y)
# log(b,a)
def my_log_yx(x,y):
    return math.log(y,x)
# a^b
def my_exp_xy(x,y):
    return x**y
# b^a
def my_exp_yx(x,y):
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

def rich_feat(data):
    columns_list = data.columns
    for i in range(0,len(columns_list)-1):
        for j in range(i+1,len(columns_list)):
            # print(columns_list[i],columns_list[j])
            V1 = columns_list[i]
            V2 = columns_list[j]
            for k in operat_list:
                new_feat_df = pd.DataFrame(rich_baisc_math(data[V1], data[V2], operat_list[k]),columns=['%s_%s_%s' % (V1,V2,k) ])
                data = pd.concat([data, new_feat_df], axis=1)
                print(i,j,k)
        # print(i)
    return data

operat_list = {
        'ADD':my_add,
        'SUB':my_sub,
        'MUL':my_multi,
        'DIV':my_division,
        # 'LOG_XY':my_log_xy,
        # 'LOG_YX':my_log_yx,
        # 'EXP_XY':my_exp_xy,
        # 'EXP_YX':my_exp_yx,
        'AVG':my_avg
    }

def rich_feat_tow(data):
    data_np = data.values
    new_columns_one = []
    new_columns_tow = []
    for ele in data_np:
        new_columns_one.append(my_variance(ele))
        new_columns_tow.append(my_avg_list(ele))
    return new_columns_one, new_columns_tow

if __name__ == '__main__':
    # print('train')
    # train_agg_data = pd.read_csv(train_agg_path)
    # train_agg_data.pop('USRID')
    # train_new_agg_data = rich_feat(train_agg_data)
    # train_new_agg_data.to_csv(train_new_agg_path,index=0)

    # print('test')
    # test_agg_data = pd.read_csv(test_agg_path)
    # test_agg_data.pop('USRID')
    # test_new_agg_data = rich_feat(test_agg_data)
    # test_new_agg_data.to_csv(test_new_agg_path,index=0)
    
    print('train')
    train_agg_data = pd.read_csv(train_agg_path)
    train_usrid = train_agg_data['USRID']
    train_agg_data.pop('USRID')
    new_one, new_tow = rich_feat_tow(train_agg_data)
    train_new_agg_data = pd.concat([train_usrid, pd.DataFrame(new_one, columns=['VARIANCE']), pd.DataFrame(new_tow, columns=['AVG_LIST'])], axis=1)
    train_new_agg_data.to_csv(train_new_agg_path,index=0)

    print('test')
    test_agg_data = pd.read_csv(test_agg_path)
    test_usrid = test_agg_data['USRID']
    test_agg_data.pop('USRID')
    new_one, new_tow = rich_feat_tow(test_agg_data)
    test_new_agg_data = pd.concat([test_usrid, pd.DataFrame(new_one, columns=['VARIANCE']), pd.DataFrame(new_tow, columns=['AVG_LIST'])], axis=1)
    test_new_agg_data.to_csv(test_new_agg_path,index=0)