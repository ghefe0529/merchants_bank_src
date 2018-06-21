# -*- coding: utf-8 -*-
# @Date    : 2018-06-20 14:23:29
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 合并log和flg

import numpy as np
import pandas as pd


common_path = r'~/Documents/Study/Python/merchants_bank'

# train
train_log_evt3_usrid_path = common_path + r'/data/feature/train_usrid_evt3.csv'

train_flg_path = common_path + r'/data/corpus/output/train_flg.csv'

train_usrid_evt3_flg_path = common_path + r'/data/feature/train_usrid_evt3_flg.csv'

def merge_etv3_flg(log_path, flg_path, save_path):
    log_df = pd.read_csv(log_path)
    flg_df = pd.read_csv(flg_path)
    combat_df = pd.merge(log_df, flg_df, how='left', on='USRID')
    combat_df.to_csv(save_path,index=0)
    print('合并完成')

if __name__ == '__main__':
    merge_etv3_flg(train_log_evt3_usrid_path, train_flg_path, train_usrid_evt3_flg_path)

    print('结束')