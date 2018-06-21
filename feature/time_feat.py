# -*- coding: utf-8 -*-
# @Date    : 2018-06-21 14:04:36
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 第一次点击的世界,最后一次点击时间,平均间隔时间

import numpy as np
import pandas as pd
import time

common_path = r'~/Documents/Study/Python/merchants_bank'
# train
#  input 
train_log_path = common_path + r'/data/corpus/output/train_log.csv'
# output
train_time_path = common_path + r'/data/feature/train_time.csv'

# test
# input
test_log_path = common_path + r'/data/corpus/output/test_log.csv'
# output
test_time_path = common_path + r'/data/feature/test_time.csv'

def avg_click_time(time_list):
    return (max(time_list) - min(time_list)/len(time_list))

# 根据usrid合并数据 
def combat_data_by_usrid(data_path, save_path):
    df = pd.read_csv(data_path)
    # 建立一个空的usrid的list
    usrid_data =[]
    # 建立一个空的第一点击时间的list
    first_tim = []
    # 建立一个空的最后一次点击时间的list
    last_tim =[]
    # 建立一个空的平均间隔时间的list
    avg_tim =[]
    for usrid, group in df.groupby(['USRID']):
        usrid_data.append(usrid)
        sum_tim = list(group['OCC_TIM'])
        sum_tim = [ (1522512000.0 - time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))) for x in sum_tim]
        # print(sum_tim)
        first_tim.append(min(sum_tim))
        last_tim.append(max(sum_tim))
        avg_tim.append(avg_click_time(sum_tim))
        # break
    time_feat_df = pd.DataFrame([])
    time_feat_df['USRID'] = usrid_data
    time_feat_df['first_tim'] = first_tim
    time_feat_df['last_tim'] = last_tim
    time_feat_df['avg_tim'] = avg_tim
    # print(time_feat_df)
    time_feat_df.to_csv(save_path,index=0)
    print('-------保存成功--------')

if __name__ == '__main__':
    combat_data_by_usrid(train_log_path, train_time_path)
    combat_data_by_usrid(test_log_path, test_time_path)