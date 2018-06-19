# -*- coding: utf-8 -*-
# @Date    : 2018-06-18 15:56:16
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 根据是否有log将usr分为两部分

import numpy as np
import pandas as pd


common_path = r'~/Documents/Study/Python/merchants_bank/'

train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'

train_log_evt3_usrid_path = common_path + r'/data/feature/train_usrid_evt3.csv'

train_log_evt3_usrid_agg_path = common_path + r'/data/feature/train_usrid_evt3_agg.csv'

if __name__ == '__main__':
    train_agg_df = pd.read_csv(train_agg_path)
    # print('train_agg_df', train_agg_df.shape)
    train_log_evt3_df = pd.read_csv(train_log_evt3_usrid_path)
    # print('train_log_evt3_df', train_log_evt3_df.shape)
    train_agg_evt3 = pd.merge(train_log_evt3_df, train_agg_df, on='USRID', how='left')
    # print('train_agg_evt3',train_agg_evt3.shape)
    train_agg_evt3.to_csv(train_log_evt3_usrid_agg_path,index=0)