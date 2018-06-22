# -*- coding: utf-8 -*-
# @Date    : 2018-06-20 14:46:02
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 计算evt3中每个类的概率，根据概率合并evt3特征


import numpy as np
import pandas as pd

# common_path = r'~/Documents/Study/Python/merchants_bank'
common_path = r'~/Documents/merchants_bank'
# train
# input
train_usrid_evt3_path = common_path + r'/data/feature/train_usrid_evt3.csv'

# train_usrid_evt3_flg_path = common_path + r'/data/feature/train_usrid_evt3_flg.csv'
# output
train_usrid_merge_evt3_path = common_path + r'/data/feature/train_usrid_merge_evt3.csv'

# test
# input
test_usrid_evt3_path = common_path + r'/data/feature/test_usrid_evt3.csv'

# test_usrid_evt3_flg_path = common_path + r'/data/feature/test_usrid_evt3_flg.csv'
# output
test_usrid_merge_evt3_path = common_path + r'/data/feature/test_usrid_merge_evt3.csv'

temp_path = common_path + r'/data/feature/temp.csv'
def count_evt3_proba():
    # 计算evt3的权重
    sum_evt3_count = 0
    train_evt3_df = pd.read_csv(train_usrid_evt3_path)
    test_evt3_df = pd.read_csv(test_usrid_evt3_path)
    # df = pd.read_csv(train_usrid_evt3_flg_path)
    both_evt3_df = pd.concat([train_evt3_df, test_evt3_df], axis=0)
    both_evt3_df.pop('USRID')
    for ele in both_evt3_df.columns:
        # print(ele)
        sum_evt3_count += sum(list(both_evt3_df[ele]))
        # print(sum_evt3_count)
        # break
    print(sum_evt3_count)

    evt3_proba = []
    # count = 0
    for ele in both_evt3_df.columns:
        evt3_count = sum(list(both_evt3_df[ele]))
        ele_evt3_proba = evt3_count/sum_evt3_count
        # if ele_evt3_proba < 0.00001:
        #     print(count)
        #     count += 1
        #     print(ele_evt3_proba)
        #     ele_evt3_proba = 0
        evt3_proba.append(ele_evt3_proba)
    
    # print('len evt3_proba', len(evt3_proba))
    # evt3_proba_df = pd.DataFrame(evt3_proba)
    # print('evt3_proba ',evt3_proba_df[:10])
    # evt3_proba_df.to_csv(temp_path)
    return evt3_proba

def sum_evt3_def(usrid_evt3_path, save_path):
    df = pd.read_csv(usrid_evt3_path)
    # print('df', df.shape)
    df_usrid = df['USRID']
    df.pop('USRID')
    print('df shape is ', df.shape)
    evt3_proba = count_evt3_proba()
    print('权值计算成功')
    merge_evt3 = []
    for ele in df.as_matrix():
        sum_evt3 = 0
        for x,j in zip(ele, evt3_proba):
            sum_evt3 += x * j
        merge_evt3.append(sum_evt3)
    # print(merge_evt3[:10])
    merge_evt3_df = pd.DataFrame(merge_evt3, columns=['MERGE_EVT3'])
    merge_evt3_df = pd.concat([df_usrid, merge_evt3_df],axis=1)
    merge_evt3_df.to_csv(save_path, index=0)
    print('保存成功')

if __name__ == '__main__':
    # count_evt3_proba()
    sum_evt3_def(train_usrid_evt3_path, train_usrid_merge_evt3_path)
    sum_evt3_def(test_usrid_evt3_path, test_usrid_merge_evt3_path)
    print('结束')
    
    