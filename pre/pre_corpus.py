# -*- coding: utf-8 -*-
# @Date    : 2018-06-18 09:40:12
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 预处理数据,将所有原始文件按照USRID升序排序

import pandas as pd

# common_path = r'~/Documents/Study/Python/merchants_bank'
common_path = r'~/Documents/merchants_bank'

# train 原始文件
train_agg_path = common_path + r'/data/corpus/train_agg.csv'
train_log_path = common_path + r'/data/corpus/train_log.csv'
train_flg_path = common_path + r'/data/corpus/train_flg.csv'

# test 原始文件
test_agg_path = common_path + r'/data/corpus/test_agg.csv'
test_log_path = common_path + r'/data/corpus/test_log.csv'

# train 排序后的文件
train_agg_out_path = common_path + r'/data/corpus/output/train_agg.csv'
train_log_out_path = common_path + r'/data/corpus/output/train_log.csv'
train_flg_out_path = common_path + r'/data/corpus/output/train_flg.csv'

# test 排序后的文件
test_agg_out_path = common_path + r'/data/corpus/output/test_agg.csv'
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