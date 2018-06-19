# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 14:53:22
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 将ev3向量化

import pandas as pd
import numpy as np


common_path = r'~/Documents/Study/Python/merchants_bank/'

train_log_path = common_path + r'/data/feature/train_log_first.csv'

train_log_evt3_usrid_path = common_path + r'/data/feature/train_usrid_evt3.csv'

def get_evt3_featrue_name(size):
    names = []
    for i in range(size):
        names.append('evt3_'+str(i))
    return names

def combat_data_by_usrid(train_log_data):
    sum_evt3_data = []
    sum_usrid_data =[]
    for usrid, group in train_log_data.groupby(['USRID']):
        sum_usrid_data.append(usrid)
        evt3_vec_list = evt3_to_vec(list(group['EVT_LBL_3'])).tolist()
        sum_evt3_data.append(evt3_vec_list)
        # break
    evt3_columns = get_evt3_featrue_name(595)
    sum_evt3_data = pd.DataFrame(sum_evt3_data,columns=evt3_columns)
    print(sum_evt3_data.shape)
    sum_usrid_data = pd.DataFrame(sum_usrid_data,columns='USRID')
    print(sum_usrid_data.shape)
    evt3_usrid_data = pd.concat([sum_usrid_data,sum_evt3_data], axis=1)
    # sum_evt3_data.to_csv(train_log_evt3_usrid_path, index=0)
    evt3_usrid_data.to_csv(train_log_evt3_usrid_path, index=0)
    print('-------保存成功--------')

train_log_data = pd.read_csv(train_log_path)
train_log_evt3_data = train_log_data['EVT_LBL_3'].as_matrix()
train_log_evt3_data = set(train_log_evt3_data)
vacablary_evt3 = [x for x in train_log_evt3_data]
vacablary_evt3.sort()

def evt3_to_vec(evt3_list):
    evt3_vec = np.zeros(595,dtype=np.int16)
    for ele in evt3_list:
        index = vacablary_evt3.index(ele)
        evt3_vec[index] += 1
        # print(evt3_vec)
    return evt3_vec

if __name__ =='__main__':
    df_log = pd.read_csv(train_log_path)
    combat_data_by_usrid(df_log)