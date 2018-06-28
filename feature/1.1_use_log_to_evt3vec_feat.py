# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 14:53:22
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 将ev3向量化(train和test)

import pandas as pd
import numpy as np


# common_path = r'~/Documents/Study/Python/merchants_bank'
common_path = r'~/Documents/merchants_bank'
# train
#  输入 
train_log_path = common_path + r'/data/corpus/output/train_log.csv'
# 输出
train_log_evt_usrid_path = common_path + r'/data/feature/1.1_train_usrid_evt.csv'

# test
# 输入
test_log_path = common_path + r'/data/corpus/output/test_log.csv'
# 输出
test_log_evt_usrid_path = common_path + r'/data/feature/1.1_test_usrid_evt.csv'

# 根据特征个数生成特征列名,作为输出csv文件的头
# 输入：size，输出：size个特征名
def get_evt_featrue_name(size):
    print('正在生成特征列名')
    names = []
    for i in range(size):
        names.append('evt_'+str(i))
    return names

# evt的第三列转化成向量
# 输入每位用户点击的evt模块，输出以evt字典大小的一维向量
def evt_to_vec(evt_list, vacablary_evt):
    evt_vec = np.zeros(len(vacablary_evt),dtype=np.int16) #生成一个以evt字典大小的一维向量并填充为0
    # 循环遍历用户点击的模块，如果已经点击则加一
    for ele in evt_list:
        index = vacablary_evt.index(ele)
        evt_vec[index] += 1
    return evt_vec

# 获取evt中最后一个模块的编号
# 输入文件地址，输出压缩后的evt
def get_evt_by_path(path):
    print('读取文件', path)
    df = pd.read_csv(path)['EVT_LBL'].values
    result = [ x.split('-')[2] for x in df ]
    return result

# 获取evt字典
# 输出evt字典
def get_vacablary_evt():
     # 获取训练集中的log_evt
    train_log_evt_data = get_evt_by_path(train_log_path)
    # 获取测试集中的log_evt
    test_log_evt_data = get_evt_by_path(test_log_path)
    print('正在生成evt字典')
    # 生成一个包含所有evt的list,为特征化向量做准备
    train_log_evt_data.extend(test_log_evt_data)
    all_log_evt_data = train_log_evt_data
    # 用set去重
    all_log_evt_data = set(all_log_evt_data)
    # 转为list并排序
    vacablary_evt = [int(x) for x in all_log_evt_data]
    vacablary_evt.sort()

    return vacablary_evt

# 将EVT3的第三列提取出然后向量话，并与USRID合并
# 输入evt字典，读取文件，保存文件
def combat_data_by_usrid(vacablary_evt, data_path, save_path):
    print('正在将evt向量化')
    # 读取log文件
    df = pd.read_csv(data_path)
    # 建立一个空的evt的list
    sum_evt_data = []
    # 建立一个空的usrid的list
    sum_usrid_data =[]
    for usrid, group in df.groupby(['USRID']):
        sum_usrid_data.append(usrid)
        evt_list_tmp = list(group['EVT_LBL'])
        evt_list_tmp = [ int(x.split('-')[2]) for x in evt_list_tmp ]
        evt_vec_list = evt_to_vec(evt_list_tmp, vacablary_evt).tolist()
        sum_evt_data.append(evt_vec_list)

    # 生成evt的列名
    evt_columns = get_evt_featrue_name(len(vacablary_evt))

    # 将list(evt)和list(usrid)转为dataframe
    sum_evt_data = pd.DataFrame(sum_evt_data, columns=evt_columns)
    sum_usrid_data = pd.DataFrame(sum_usrid_data, columns=['USRID'])
    # 合并evt和usrid并保存
    evt_usrid_data = pd.concat([sum_usrid_data,sum_evt_data], axis=1)
    evt_usrid_data.to_csv(save_path, index=0)
    print('保存成功')

if __name__ =='__main__':
    vacablary = get_vacablary_evt()
    combat_data_by_usrid(vacablary, train_log_path, train_log_evt_usrid_path)
    combat_data_by_usrid(vacablary, test_log_path, test_log_evt_usrid_path)
    print('结束')
