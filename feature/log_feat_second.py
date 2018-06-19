# -*- coding: utf-8 -*-
# @Date    : 2018-06-18 14:41:30
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 将log文件中按照userid分组合并

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


common_path = r'~/Documents/Study/Python/merchants_bank/'

train_log_path = common_path + r'/data/feature/train_log_first.csv'
train_flg_path = common_path + r'data/corpus/output/train_flg.csv'

train_log_feature_path = common_path + r'/data/feature/train_log_second.csv'

train_log_flg_path = common_path + r'/data/feature/train_log_flg.csv'
train_only_log_flg_path = common_path + r'/data/feature/train_only_log_flg.csv'



def combat_data_by_usrid(train_log_data):
    train_log_handle_data = []
    for usrid, group in train_log_data.groupby(['USRID']):
        tmp = dict()
        tmp['USRID'] = usrid
        # print(type(group))
        tmp['OCC_TIM'] = list(group['OCC_TIM'])
        tmp['TCH_TYP'] = list(group['TCH_TYP'])
        # tmp['EVT_LBL_1'] = list(group['EVT_LBL_1'])
        # tmp['EVT_LBL_2'] = list(group['EVT_LBL_2'])
        tmp['EVT_LBL_VEC'] = evt3_to_vec(list(group['EVT_LBL_3'])).tolist()
        train_log_handle_data.append(tmp)
        # break
    train_log_handle_data = pd.DataFrame(train_log_handle_data, columns=['USRID','OCC_TIM','TCH_TYP','EVT_LBL_VEC'])
    train_log_handle_data.to_csv(train_log_feature_path,index=0)
    print('-------保存成功--------')

# 将与USRID相关的flag加入进去
def flag_to_log(train_log_data, train_flg_data):
    # print(train_log_data)
    train_result_data = pd.merge(train_log_data, train_flg_data, on='USRID')
    # print(train_result_data)
    return train_result_data

# 建立包含所有EVT3的列表
def evt3_to_vec(evt3_list):
    train_log_data = pd.read_csv(train_log_path)
    train_log_evt3_data = train_log_data['EVT_LBL_3'].as_matrix()
    train_log_evt3_data = set(train_log_evt3_data)
    vacablary_evt3 = [x for x in train_log_evt3_data]
    vacablary_evt3.sort()
    for ele in evt3_list:
        evt3_vec = np.zeros(595,dtype=np.int16)
        index = vacablary_evt3.index(ele)
        evt3_vec[index] += 1
        # print(evt3_vec)
    return evt3_vec


if __name__ == '__main__':
    train_log_data = pd.read_csv(train_log_path)

    # train_log_evt1_data = train_log_data['EVT_LBL_1'].as_matrix()
    # train_log_evt2_data = train_log_data['EVT_LBL_2'].as_matrix()
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
    
    # print(len(train_log_evt3_data))
    # 595


    # 按USRID合并数据
    combat_data_by_usrid(train_log_data)


    # train_log_data = pd.read_csv(train_log_feature_path)
    # print(train_log_data)


    # 将与USRID相关的flag加入进去
    # train_flg_data = pd.read_csv(train_flg_path)
    # train_log_data = flag_to_log(train_log_data, train_flg_data)
    # train_log_data.to_csv(train_log_flg_path, index=0)
    # train_log_data.to_csv(train_only_log_flg_path, index=0, columns=['USRID', 'FLAG'])

    # 绘制热力图，查看一下变量之间的关系
    # corrmat = train_log_data.corr('spearman')
    # f, ax = plt.subplots(figsize=(12, 9))
    # ax.set_xticklabels(corrmat, rotation='horizontal')
    # sns.heatmap(np.fabs(corrmat), square=False, center=1)
    # label_y = ax.get_yticklabels()
    # plt.setp(label_y, rotation=360)
    # label_x = ax.get_xticklabels()
    # plt.setp(label_x, rotation=90)
    # plt.show()
