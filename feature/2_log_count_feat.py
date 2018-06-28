# -*- coding: utf-8 -*-
# @Date    : 2018-06-21 14:04:36
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 统计用户点击log的次数

import numpy as np
import pandas as pd


# common_path = r'~/Documents/Study/Python/merchants_bank'
common_path = r'~/Documents/merchants_bank'
# train
#  input 
train_log_path = common_path + r'/data/corpus/output/train_log.csv'
# output
train_log_count_path = common_path + r'/data/feature/2_train_log_count.csv'

# test
# input
test_log_path = common_path + r'/data/corpus/output/test_log.csv'
# output
test_log_count_path = common_path + r'/data/feature/2_test_log_count.csv'

# 根据usrid合并数据 
def combat_data_by_usrid(data_path, save_path):
    print('读取文件', data_path)
    df = pd.read_csv(data_path)
    # 计算每位用户的点击次数和点击了几种模块
    log_count_df= df.groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVT_LEN':len,'EVT_SET_LEN':lambda x: len(set(x))})
    # 保持文件
    log_count_df.to_csv(save_path,index=0)
    print('保存成功')

if __name__ == '__main__':
    combat_data_by_usrid(train_log_path, train_log_count_path)
    combat_data_by_usrid(test_log_path, test_log_count_path)
    print('结束')