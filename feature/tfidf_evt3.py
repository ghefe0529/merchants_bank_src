# -*- coding: utf-8 -*-
# @Date    : 2018-06-23 13:45:11
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 用tfidf特征化evt3


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# common_path = r'~/Documents/Study/Python/merchants_bank'
common_path = r'~/Documents/merchants_bank'

# train
#  input 
train_log_path = common_path + r'/data/corpus/output/train_log.csv'

train_flg_path = common_path + r'/data/corpus/output/train_flg.csv'

# output
train_tfidf_path = common_path + r'/data/feature/train_tfidf.csv'

# test
# input
test_log_path = common_path + r'/data/corpus/output/test_log.csv'

# output
test_tfidf_path = common_path + r'/data/feature/test_tfidf.csv'


# 堆叠模型的保存
lr_model_path = common_path + r'/data/model/LR_model.m'
svm_model_path = common_path + r'/data/model/svm_model.m'
randomf_model_path = common_path + r'/data/model/randomf_model.m'
bayes_model_path = common_path + r'/data/model/bayes_model.m'


def handle_evt3(data):
    # 存储usrid
    usrid_list = []
    # 存储每个usrid的所有evt3
    evt3_list = []
    for usrid, group in data.groupby('USRID'):
        usrid_list.append(usrid)
        evt3_list.append(' '.join(list(group['EVT_LBL'])))
        # break
    # print(usrid_list)
    # print(evt3_list)
    usrid_df = pd.DataFrame(usrid_list, columns=['USRID'])
    evt3_df = pd.DataFrame(evt3_list, columns=['EVT_LIST'])
    return pd.concat([usrid_df,evt3_df],axis=1)

def tfidf_handle_data(data):
    tfv = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf=True, stop_words='english')
    tfv.fit(data)
    result = tfv.transform(data)
    return result

if __name__ == '__main__':
    # 获取训练集的数据
    print('train')
    train_log_data = pd.read_csv(train_log_path)
    train_evt_list_data = handle_evt3(train_log_data)
    # 获取测试集的数据
    print('test')
    test_log_data = pd.read_csv(test_log_path)
    test_evt_list_data = handle_evt3(test_log_data)

    print(train_evt_list_data.shape)
    train_n = train_evt_list_data.shape[0]
    # 合并训练集和测试集
    evt_list_data = pd.concat([train_evt_list_data,test_evt_list_data],axis=0)
    # 通过tfidf对evt特征化
    tfidf_data = tfidf_handle_data(evt_list_data['EVT_LIST'])

    # '''
    tfidf_data_df = pd.DataFrame(tfidf_data.todense())
    usrid_df = pd.DataFrame(evt_list_data['USRID'],columns=['USRID'])
    print('evt_list_data[USRID]', usrid_df.shape)
    print('tfidf_data_df', tfidf_data_df.shape)

    # tfidf_data_df = tfidf_data_df.loc[~tfidf_data_df.index.duplicated(keep='first')]
    tfidf_data_df = tfidf_data_df.reset_index(drop=True) 
    usrid_df = usrid_df.reset_index(drop=True) 

    # tfidf_data_df['USRID'] = evt_list_data['USRID']
    # usrid_df = usrid_df.loc[~usrid_df.index.duplicated(keep='first')]
    # tfidf_data_df = tfidf_data_df.loc[~tfidf_data_df.index.duplicated(keep='first')]

    tfidf_data_df = pd.concat([usrid_df, tfidf_data_df], axis=1)

    tfidf_data_df[:train_n].to_csv(train_tfidf_path,index=0)
    tfidf_data_df[train_n:].to_csv(test_tfidf_path,index=0)
    # '''
    