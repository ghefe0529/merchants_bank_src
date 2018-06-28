# -*- coding: utf-8 -*-
# @Date    : 2018-06-21 14:04:36
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 第一次点击的世界,最后一次点击时间,平均间隔时间

import numpy as np
import pandas as pd
import time

# common_path = r'~/Documents/Study/Python/merchants_bank'
common_path = r'~/Documents/merchants_bank'
# train
# 输入
train_log_path = common_path + r'/data/corpus/output/train_log.csv'
# 输出
train_time_path = common_path + r'/data/feature/3_train_time.csv'

# test
# 输入
test_log_path = common_path + r'/data/corpus/output/test_log.csv'
# 输出
test_time_path = common_path + r'/data/feature/3_test_time.csv'

# 根据usrid合并数据 
# 输入读取文件
def combat_data_by_usrid(data_path, save_path):
    print('正在读取文件 ', data_path)
    df = pd.read_csv(data_path)
    # 建立一个空的usrid的list
    usrid_data =[]
    # 时间间隔内平均点击次数
    avg_tim =[]
    # 最后一次点击时间
    last_click = []

    for usrid, group in df.groupby(['USRID']):
        usrid_data.append(usrid)
        sum_tim = list(group['OCC_TIM'])
        # 最后一次点击时间
        last_click.append(min([31-int(x[8:10]) for x in sum_tim]))

        sum_tim = [ (1522512000.0 - time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))) for x in sum_tim]
        sum_tim.sort(reverse=True)

        # 每个单位时间（一个小时）内的点击次数
        interval_time =[]
        # 基础时间
        basic_time = max(sum_tim)
        # 单位时间点击次数
        count = 0

        for x in sum_tim:
            if x > (basic_time-3600):
                count += 1
            else:
                interval_time.append(count)
                basic_time = x
                count = 1

        interval_time.append(count)
        avg_tim.append(sum(interval_time)/len(interval_time))

    # 保存文件 
    time_feat_df = pd.DataFrame([])
    time_feat_df['USRID'] = usrid_data
    time_feat_df['AVG_TIM'] = avg_tim
    time_feat_df['LAST_TIM'] = last_click

    time_feat_df.to_csv(save_path,index=0)

    print('保存成功')

if __name__ == '__main__':
    combat_data_by_usrid(train_log_path, train_time_path)
    combat_data_by_usrid(test_log_path, test_time_path)
    print('结束')