# -*- coding: utf-8 -*-
# @Date    : 2018-06-18 14:41:30
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 将log文件中按照userid分组合并

import numpy as np
import pandas as pd


common_path = r'~/Documents/Study/Python/merchants_bank/'

train_log_path = common_path + r'/data/feature/train_log_first.csv'

train_log_feature_path = common_path + r'/data/feature/train_log_second.csv'


def combat_data_by_usrid(train_log_data):
    train_log_handle_data = []
    for usrid, group in train_log_data.groupby(['USRID']):
        tmp = dict()
        tmp['USRID'] = usrid
        # print(type(group))
        tmp['OCC_TIM'] = list(group['OCC_TIM'])
        tmp['TCH_TYP'] = list(group['TCH_TYP'])
        tmp['EVT_LBL_1'] = list(group['EVT_LBL_1'])
        tmp['EVT_LBL_2'] = list(group['EVT_LBL_2'])
        tmp['EVT_LBL_3'] = list(group['EVT_LBL_3'])
        train_log_handle_data.append(tmp)
        break
    train_log_handle_data = pd.DataFrame(train_log_handle_data, columns=['USRID','OCC_TIM','TCH_TYP','EVT_LBL_1','EVT_LBL_2','EVT_LBL_3'])
    # train_log_handle_data.to_csv(train_log_feature_path,index=0)
    print('-------保存成功--------')

if __name__ == '__main__':
    train_log_data = pd.read_csv(train_log_path)

    train_log_evt1_data = train_log_data['EVT_LBL_1'].as_matrix()
    train_log_evt2_data = train_log_data['EVT_LBL_2'].as_matrix()
    train_log_evt3_data = train_log_data['EVT_LBL_3'].as_matrix()

    # EVT1 len
    # train_log_evt1_data = set(train_log_evt1_data)
    # print(len(train_log_evt1_data))
    # 21


    # EVT2 len
    # train_log_evt2_data = set(train_log_evt2_data)
    # print(len(train_log_evt2_data))
    # 178

    # EVT3 len
    # train_log_evt3_data = set(train_log_evt3_data)
    # print(len(train_log_evt3_data))
    # 595


    # 按USRID合并数据
    combat_data_by_usrid(train_log_data)


    # train_log_data = pd.read_csv(train_log_feature_path)
    # print(train_log_data)