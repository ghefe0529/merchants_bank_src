# -*- coding: utf-8 -*-
# @Date    : 2018-06-20 14:46:02
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 计算evt3中每个类的得分，根据得分将一维的evt特征合并成数值


import numpy as np
import pandas as pd

# common_path = r'~/Documents/Study/Python/merchants_bank'
common_path = r'~/Documents/merchants_bank'
# train
# 输入
train_usrid_evt_path = common_path + r'/data/feature/1.1_train_usrid_evt.csv'

# 输出 
train_usrid_merge_evt_path = common_path + r'/data/feature/1.2_train_usrid_merge_evt.csv'

# test
# 输入
test_usrid_evt_path = common_path + r'/data/feature/1.1_test_usrid_evt.csv'

# 输出
test_usrid_merge_evt_path = common_path + r'/data/feature/1.2_test_usrid_merge_evt.csv'

# 计算evt的得分,得分计算方式：每个类的点击总次数除以所有类的点击数
# 输入每个evt的得分
def count_evt_score():
    print('正在计算evt的分数')
    # 获取train的evt 
    train_evt_df = pd.read_csv(train_usrid_evt_path)
    # 或test的evt
    test_evt_df = pd.read_csv(test_usrid_evt_path)
    # 合并test和train的evt
    both_evt_df = pd.concat([train_evt_df, test_evt_df], axis=0)
    both_evt_df.pop('USRID')

    # 所有类的点击总数
    sum_evt_count = 0

    for ele in both_evt_df.columns:
        sum_evt_count += sum(list(both_evt_df[ele]))
    
    # 每个evt的得分
    evt_score = []
    for ele in both_evt_df.columns:
        evt_count = sum(list(both_evt_df[ele]))
        ele_evt_score = evt_count/sum_evt_count
        evt_score.append(ele_evt_score)
    
    print('得分计算成功')
    return evt_score

# 根据得分将一维的evt特征合并成数值
# 输入读取已经向量化文件的路径，保存路径
def merge_evt(evt_score, usrid_evt_path, save_path):
    print('正在根据evt得分压缩evt向量')
    # 读取已经向量化的文件
    df = pd.read_csv(usrid_evt_path)
    # 暂存usrid
    df_usrid = df['USRID']
    df.pop('USRID')
    # evt合并后数值的list
    merge_evt = []
    for ele in df.values:
        usrid_evt_score = 0
        for x,j in zip(ele, evt_score):
            usrid_evt_score += x * j
        merge_evt.append(usrid_evt_score)
    # 转为dataframe保存
    merge_evt_df = pd.DataFrame(merge_evt, columns=['MERGE_EVT3'])
    merge_evt_df = pd.concat([df_usrid, merge_evt_df],axis=1)
    merge_evt_df.to_csv(save_path, index=0)
    print('保存成功')

if __name__ == '__main__':
    evt_score = count_evt_score()
    merge_evt(evt_score, train_usrid_evt_path, train_usrid_merge_evt_path)
    merge_evt(evt_score, test_usrid_evt_path, test_usrid_merge_evt_path)
    print('结束')
    
    