# -*- coding: utf-8 -*-
# @Date    : 2018-06-18 10:12:01
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 处理log文件的特征

import pandas as pd
import numpy as np
import time

common_path = r'~/Documents/Study/Python/merchants_bank/'

# train corpus data path
train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'
train_log_path = common_path + r'/data/corpus/output/train_log.csv'
train_flg_path = common_path + r'data/corpus/output/train_flg.csv'

# train feature log path
train_log_feature_path = common_path + r'/data/feature/train_log_first.csv'

def flg_sum(train_flg_data, train_log_usrid_data):
    # 查看这些用户的购买标记
    # 有test flg_1 2860 sum 39028
    # 无test flg_1 316 sum 40972
    flg_1 = 0
    sum = 0
    for usrid,flg in zip(train_flg_data['USRID'], train_flg_data['FLAG']):
        if usrid not in train_log_usrid_data:
            sum += 1
            if flg == 1:
                flg_1 += 1
    print('sum is ', sum,'flg_1 is', flg_1)

# 处理log时间列
def handle_log_time(train_log_data):
    train_log_time_data = train_log_data['OCC_TIM'].as_matrix()
    # print('type(train_log_time_data', type(train_log_time_data[0]))
    # print('train_log_time_data[0]', train_log_time_data[0])
    # print('train_log_time_data[0]', time.mktime(time.strptime(train_log_time_data[0], '%Y-%m-%d %H:%M:%S')))
    train_log_time_data = [ int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))) for x in train_log_time_data ]
    # print(train_log_time_data[:10])
    train_log_data['OCC_TIM'] = pd.DataFrame(train_log_time_data)
    return train_log_data
    # print(train_log_data[:10])

def handle_log_evt(train_log_data):
    train_log_evt_data = train_log_data['EVT_LBL'].as_matrix()
    # print(train_log_evt_data)
    train_log_evt_data = [x.split('-') for x in train_log_evt_data]
    train_log_evt_data = pd.DataFrame(train_log_evt_data, columns=['1','2','3'])
    # print(train_log_evt_data)
    train_log_data['EVT_LBL_1'] = train_log_evt_data['1']
    train_log_data['EVT_LBL_2'] = train_log_evt_data['2']
    train_log_data['EVT_LBL_3'] = train_log_evt_data['3']
    train_log_data = train_log_data.drop(['EVT_LBL'], axis=1)
    return train_log_data

if __name__ == '__main__':
    # 获取train log
    train_log_data = pd.read_csv(train_log_path)
    train_flg_data = pd.read_csv(train_flg_path)
    # 获取有log记录的USRID
    train_log_usrid_data = set(train_log_data['USRID'].as_matrix())
    # print('len train_log_data', len(train_log_data))
    # 3533818
    # print('len train_log_usrid_data is ',len(train_log_usrid_data))
    # 39028
    # print('type(train_log_usrid_data) ',type(train_log_usrid_data))

    # 获取train_flg文件内的内容
    # train_flg_data = pd.read_csv(train_flg_path, train_log_usrid_data)
    
    # 查看这些用户的购买标记
    flg_sum(train_flg_data, train_log_usrid_data)

    # 处理log内的时间特征
    # print('处理log内的时间特征')
    # train_log_data = handle_log_time(train_log_data)

    # 处理log内的EVT_LBL特征
    # print('处理EVT_LBL')
    # train_log_data = handle_log_evt(train_log_data)
    # 将处理了时间和evt后的log写入文件
    # train_log_data.to_csv(train_log_feature_path, index=0)