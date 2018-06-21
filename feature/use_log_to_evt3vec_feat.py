# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 14:53:22
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 将ev3向量化(train和test)

import pandas as pd
import numpy as np


common_path = r'~/Documents/Study/Python/merchants_bank'
# train
#  input 
train_log_path = common_path + r'/data/corpus/output/train_log.csv'
# output
train_log_evt3_usrid_path = common_path + r'/data/feature/train_usrid_evt3.csv'

# test
# input
test_log_path = common_path + r'/data/corpus/output/test_log.csv'
# output
test_log_evt3_usrid_path = common_path + r'/data/feature/test_usrid_evt3.csv'

# 根据特征个数生成特征列名
def get_evt3_featrue_name(size):
    print('生成特征列名')
    names = []
    for i in range(size):
        names.append('evt3_'+str(i))
    return names

# evt3的第三列转化成向量
def evt3_to_vec(evt3_list, vacablary_evt3):
    evt3_vec = np.zeros(len(vacablary_evt3),dtype=np.int16)
    for ele in evt3_list:
        index = vacablary_evt3.index(ele)
        evt3_vec[index] += 1
    return evt3_vec

def get_evt3_by_path(path):
    df = pd.read_csv(path)['EVT_LBL'].as_matrix()
    result = [ x.split('-')[2] for x in df ]
    return result

def get_vacablary_evt3():
     # 获取训练集中的log
    train_log_evt3_data = get_evt3_by_path(train_log_path)
    print('train len ',len(train_log_evt3_data))
    print('train type ',type(train_log_evt3_data))
    # 获取测试集中的log
    test_log_evt3_data = get_evt3_by_path(test_log_path)
    print('test len ', len(test_log_evt3_data))
    print('test type ', type(test_log_evt3_data))

    # 生成一个包含所有evt3的list为特征化向量做准备
    train_log_evt3_data.extend(test_log_evt3_data)
    all_log_evt3_data = train_log_evt3_data
    # print(all_log_evt3_data)
    # all_log_evt3_data = train_log_evt3_data
    all_log_evt3_data = set(all_log_evt3_data)
    vacablary_evt3 = [int(x) for x in all_log_evt3_data]
    vacablary_evt3.sort()
    print(len(vacablary_evt3))
    return vacablary_evt3

# 将EVT3的第三列提取出来与usrid合并
def combat_data_by_usrid(vacablary_evt3, data_path, save_path):
    df = pd.read_csv(data_path)
    # 建立一个空的evt3的list
    sum_evt3_data = []
    # 建立一个空的usrid的list
    sum_usrid_data =[]
    for usrid, group in df.groupby(['USRID']):
        sum_usrid_data.append(usrid)
        evt3_list_tmp = list(group['EVT_LBL'])
        evt3_list_tmp = [ int(x.split('-')[2]) for x in evt3_list_tmp ]
        evt3_vec_list = evt3_to_vec(evt3_list_tmp, vacablary_evt3).tolist()
        sum_evt3_data.append(evt3_vec_list)
        # break
    # 生成evt3的列名
    evt3_columns = get_evt3_featrue_name(len(vacablary_evt3))
    # 将list(evt3)和list(usrid)转为dataframe
    sum_evt3_data = pd.DataFrame(sum_evt3_data, columns=evt3_columns)
    print(sum_evt3_data.shape)
    sum_usrid_data = pd.DataFrame(sum_usrid_data, columns=['USRID'])
    print(sum_usrid_data.shape)
    # 合并evt3和usrid并保存
    evt3_usrid_data = pd.concat([sum_usrid_data,sum_evt3_data], axis=1)
    evt3_usrid_data.to_csv(save_path, index=0)
    print('-------保存成功--------')

if __name__ =='__main__':
    vacablary = get_vacablary_evt3()
    combat_data_by_usrid(vacablary,train_log_path,train_log_evt3_usrid_path)
    combat_data_by_usrid(vacablary,test_log_path,test_log_evt3_usrid_path)
    # get_evt3_by_path(train_log_path)