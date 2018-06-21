# -*- coding: utf-8 -*-
# @Date    : 2018-06-20 10:57:30
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 将没有log的usrid分离开

import numpy as np
import pandas as pd


common_path = r'~/Documents/Study/Python/merchants_bank'
# train
train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'

train_log_path = common_path + r'/data/corpus/output/train_log.csv'

train_agg_without_log_path = common_path + r'/data/feature/train_agg_without_log.csv'

train_agg_with_log_path = common_path + r'/data/feature/train_agg_with_log.csv'

# test
test_agg_path = common_path + r'/data/corpus/output/test_agg.csv'

test_log_path = common_path + r'/data/corpus/output/test_log.csv'

test_agg_without_log_path = common_path + r'/data/feature/test_agg_without_log.csv'

test_agg_with_log_path = common_path + r'/data/feature/test_agg_with_log.csv'


def separation_usrid(agg_path, log_path, save_withlog_path, save_withoutlog_path):
    # 将没有log的用户分离开
    agg_df = pd.read_csv(agg_path)
    log_df = pd.read_csv(log_path, usecols=[0]).as_matrix()
    print('log_df shape', log_df.shape)
    print(log_df.ravel().tolist()[:10])
    log_set = set(log_df.ravel().tolist())
    log_list = [x for x in log_set]
    drop_list = agg_df.loc[agg_df['USRID'].isin(log_list)].index
    # stay_list = list(agg_df.index) - list(drop_list)
    stay_list = list(agg_df.index)
    for drop_ele in drop_list:
        stay_list.remove(drop_ele)

    agg_without_df = agg_df.drop(drop_list)
    agg_with_df = agg_df.drop(stay_list)
    print('agg len ', agg_df.shape)
    print('agg_whit len ', agg_with_df.shape)
    print('agg_whithout len ', agg_without_df.shape)
    agg_without_df.to_csv(save_withoutlog_path,index=0)
    agg_with_df.to_csv(save_withlog_path,index=0)


if __name__ == '__main__':
    separation_usrid(train_agg_path,train_log_path, train_agg_with_log_path, train_agg_without_log_path)
    separation_usrid(test_agg_path, test_log_path, test_agg_with_log_path, test_agg_without_log_path)
