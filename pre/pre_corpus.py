# -*- coding: utf-8 -*-
# @Date    : 2018-06-18 09:40:12
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 预处理数据

import pandas as pd

common_path = r'~/Documents/Study/Python/merchants_bank/'

# train corpus data path
train_agg_path = common_path + r'/data/corpus/train_agg.csv'
train_log_path = common_path + r'/data/corpus/train_log.csv'
train_flg_path = common_path + r'data/corpus/train_flg.csv'

# test corpus data path
test_agg_path = common_path + r'data/corpus/test_agg.csv'
test_log_path = common_path + r'/data/corpus/test_log.csv'

# train out data path
train_agg_out_path = common_path + r'/data/corpus/output/train_agg.csv'
train_log_out_path = common_path + r'/data/corpus/output/train_log.csv'
train_flg_out_path = common_path + r'data/corpus/output/train_flg.csv'

# test out data path
test_agg_out_path = common_path + r'data/corpus/output/test_agg.csv'
test_log_out_path = common_path + r'/data/corpus/output/test_log.csv'

def df_sore_values(df_path, out_path):
    df = pd.read_csv(df_path, sep='\t')
    df = df.sort_values(by='USRID', ascending=True)
    df.to_csv(out_path, index=0) 

if __name__ == '__main__':
    print('write train agg ')
    df_sore_values(train_agg_path, train_agg_out_path)
    print('write train log ')
    df_sore_values(train_log_path, train_log_out_path)
    print('write train flg ')
    df_sore_values(train_flg_path, train_flg_out_path)
    print('write test agg ')
    df_sore_values(test_agg_path, test_agg_out_path)
    print('write test log ')
    df_sore_values(test_log_path, test_log_out_path)