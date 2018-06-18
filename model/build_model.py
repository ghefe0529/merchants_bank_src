# -*- coding: utf-8 -*-
# @Date    : 2018-06-17 16:26:14
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 简单模型训练训练集

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

common_path = r'~/Documents/Study/Python/merchants_bank/'

train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'
# train_log_path = r'/data/corpus/train_log.csv'
train_flg_path = common_path + r'data/corpus/output/train_flg.csv'


test_agg_path = common_path + r'data/corpus/output/test_agg.csv'
# test_log_path = r'/data/corpus/test_log.csv'

result_path = common_path + r'data/corpus/output/test_result.csv'

def pre_proba_to_csv(pre_proba):
    # 获取测试集的USRID
    test_agg_usrid_data = pd.read_csv(test_agg_path)['USRID']
    # 将概率转化为DataFrame
    result_data_pre = pd.DataFrame(pre_proba, columns=['RST'])
    # 合并概率和USRID为DataFrame
    result_data = pd.concat([test_agg_usrid_data, result_data_pre],axis=1)

    print(result_data[:10])
    # 存储结果
    result_data.to_csv(result_path, index=0, sep='\t')


if __name__ == '__main__':
    # 读取训练集
    train_agg_data = pd.read_csv(train_agg_path).as_matrix()
    train_flg_data = pd.read_csv(train_flg_path).as_matrix()
    # 读取测试集
    test_agg_data = pd.read_csv(test_agg_path).as_matrix()

    x_train = train_agg_data[:,:30]
    y_train = train_flg_data[:,1:]
    x_test = test_agg_data[:,:30]

    print('x_train shape is ',x_train.shape)
    print('x_test shape is ',x_test.shape)
    print('y_train shape is ',y_train.shape)
    # 训练模型
    lgr = LogisticRegression()
    lgr.fit(x_train, y_train.ravel())
    y_pre_proba = lgr.predict_proba(x_test)
    
    # 取出训练到概率并存入文件
    pre_proba_to_csv(y_pre_proba[:, 1:])
    