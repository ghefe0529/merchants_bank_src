# -*- coding: utf-8 -*-
# @Date    : 2018-06-18 15:56:16
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 将有log的usr将agg和merge_evt合并

import numpy as np
import pandas as pd


common_path = r'~/Documents/Study/Python/merchants_bank/'

# train
train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'

train_log_evt3_usrid_path = common_path + r'/data/feature/train_usrid_evt3.csv'

train_flg_path = common_path + r'/data/corpus/output/train_flg.csv'

train_log_evt3_usrid_agg_path = common_path + r'/data/feature/train_usrid_evt3_agg.csv'


# test
test_agg_path = common_path + r'/data/corpus/output/test_agg.csv'

test_log_evt3_usrid_path = common_path + r'/data/feature/test_usrid_evt3.csv'

test_log_evt3_usrid_agg_path = common_path + r'/data/feature/test_usrid_evt3_agg.csv'


def get_usrid_evt3_agg_flg_csv(agg_path, log_evt3_usrid_path, save_path):
    agg_df = pd.read_csv(agg_path)
    # print('train_agg_df', train_agg_df.shape)
    log_evt3_df = pd.read_csv(log_evt3_usrid_path)
    # print('train_log_evt3_df', train_log_evt3_df.shape)
    agg_log_evt3 = pd.merge(log_evt3_df, agg_df, on='USRID', how='left')
    agg_log_evt3.to_csv(save_path, index=0)
    print('保存成功')


if __name__ == '__main__':
    # 使用训练集生成usrid_evt3_agg_flg文件

    get_usrid_evt3_agg_flg_csv(train_agg_path, 
                                train_log_evt3_usrid_path, 
                                train_log_evt3_usrid_agg_path)
    '''
    # 使用测试集生存usrid_evt3_agg文件
    get_usrid_evt3_agg_flg_csv(test_agg_path, 
                                test_log_evt3_usrid_path, 
                                test_log_evt3_usrid_agg_path)
    '''
    print('完成')
