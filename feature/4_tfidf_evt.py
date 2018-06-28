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
#  输入 
train_log_path = common_path + r'/data/corpus/output/train_log.csv'

# 输出
train_tfidf_path = common_path + r'/data/feature/4_train_tfidf.csv'

# test
# 输入
test_log_path = common_path + r'/data/corpus/output/test_log.csv'

# 输出
test_tfidf_path = common_path + r'/data/feature/4_test_tfidf.csv'

# 根据特征个数生成特征列名,作为输出csv文件的头
# 输入：size，输出：size个特征名
def get_evt_featrue_name(size):
    print('正在生成特征列名')
    names = []
    for i in range(size):
        names.append('tfidf_'+str(i))
    return names

def handle_evt(data):
    evt_tfidf_df = data.groupby('USRID', as_index=False)['EVT_LBL'].agg({'TFIDF':lambda x: ' '.join(x)})
    return evt_tfidf_df

def tfidf_handle_data(data):
    print('tfidf特征化evt')
    tfv = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf=True, stop_words='english')
    tfv.fit(data)
    result = tfv.transform(data)
    return result

if __name__ == '__main__':
    # 获取训练集的数据
    print('train')
    train_log_data = pd.read_csv(train_log_path)
    train_evt_list_data = handle_evt(train_log_data)
    # 获取测试集的数据
    print('test')
    test_log_data = pd.read_csv(test_log_path)
    test_evt_list_data = handle_evt(test_log_data)

    # 记录训练个数
    train_n = train_evt_list_data.shape[0]

    # 合并训练集和测试集
    evt_list_data = pd.concat([train_evt_list_data,test_evt_list_data],axis=0, ignore_index=True)

    # 通过tfidf对evt特征化
    tfidf_data = tfidf_handle_data(evt_list_data['TFIDF'])
    # 将结果转为dataframe
    tfidf_columns = get_evt_featrue_name(tfidf_data.shape[1])
    tfidf_data_df = pd.DataFrame(tfidf_data.todense(), columns=tfidf_columns)
    # 添加USRID
    usrid_df = pd.DataFrame(evt_list_data['USRID'],columns=['USRID'])
    # 重置index否则合并时会报错
    tfidf_data_df = tfidf_data_df.reset_index(drop=True) 
    usrid_df = usrid_df.reset_index(drop=True) 
    # 合并USRID和tfidf
    tfidf_data_df = pd.concat([usrid_df, tfidf_data_df], axis=1)
    # 保存文件
    print('保存文件')
    tfidf_data_df[:train_n].to_csv(train_tfidf_path, index=0)
    tfidf_data_df[train_n:].to_csv(test_tfidf_path, index=0)
    print('结束')