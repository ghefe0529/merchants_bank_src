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
train_log_count_path = common_path + r'/data/feature/train_log_count.csv'

# test
# input
test_log_path = common_path + r'/data/corpus/output/test_log.csv'
# output
test_log_count_path = common_path + r'/data/feature/test_log_count.csv'

# 根据usrid合并数据 
def combat_data_by_usrid(data_path, save_path):
    df = pd.read_csv(data_path)
    # 建立一个空的log_count的list
    log_count = []
    # 建立一个空的usrid的list
    usrid_data =[]
    # 查看count的最大值,最小值
    max_count = 0
    min_count = 100
    for usrid, group in df.groupby(['USRID']):
        usrid_data.append(usrid)
        count = group.shape[0]
        log_count.append(count)
        if max_count < count:
            max_count = count
        if min_count > count:
            min_count = count
        # break
    log_count_df = pd.DataFrame([])
    log_count_df['USRID'] = usrid_data
    log_count_df['LOG_COUNT'] = log_count
    # print(log_count_df)
    log_count_df.to_csv(save_path,index=0)
    print('max_count is ', max_count)
    print('min_count is ', min_count)
    print('-------保存成功--------')

if __name__ == '__main__':
    combat_data_by_usrid(train_log_path, train_log_count_path)
    combat_data_by_usrid(test_log_path, test_log_count_path)