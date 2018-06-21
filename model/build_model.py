# -*- coding: utf-8 -*-
# @Date    : 2018-06-17 16:26:14
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 简单模型训练训练集

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost

common_path = r'~/Documents/Study/Python/merchants_bank/'
# train 
train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'
train_usrid_merge_evt3_path = common_path + r'/data/feature/train_usrid_merge_evt3.csv'
train_pre_usrid_merge_evt3_path = common_path + r'/data/feature/train_pre_usrid_merge_evt3.csv'

# train_log_path = r'/data/corpus/train_log.csv'
train_flg_path = common_path + r'data/corpus/output/train_flg.csv'
train_temp = common_path + r'/data/feature/trian_temp.csv'

# test
test_agg_path = common_path + r'data/corpus/output/test_agg.csv'
test_usrid_merge_evt3_path = common_path + r'/data/feature/test_usrid_merge_evt3.csv'
test_pre_usrid_merge_evt3_path = common_path + r'/data/feature/test_pre_usrid_merge_evt3.csv'
# test_log_path = r'/data/corpus/test_log.csv'
test_temp = common_path + r'/data/feature/test_temp.csv'

result_path = common_path + r'data/corpus/output/test_result1.csv'

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
    train_agg_data = pd.read_csv(train_agg_path)
    train_usrid_merge_evt3_data = pd.read_csv(train_usrid_merge_evt3_path)
    train_pre_usrid_merge_evt3_data = pd.read_csv(train_pre_usrid_merge_evt3_path)
    train_both_usrid_mergr_evt3_data = pd.concat([train_usrid_merge_evt3_data, train_pre_usrid_merge_evt3_data],axis=0)

    train_df = pd.merge(train_agg_data,train_both_usrid_mergr_evt3_data, how='left', on='USRID')
    train_df.to_csv(train_temp,index=0)

    # Y
    train_flg_data = pd.read_csv(train_flg_path)

    # 读取测试集
    test_agg_data = pd.read_csv(test_agg_path)
    test_usrid_merge_evt3_data = pd.read_csv(test_usrid_merge_evt3_path)
    test_pre_usrid_merge_evt3_data = pd.read_csv(test_pre_usrid_merge_evt3_path)
    test_both_usrid_mergr_evt3_data = pd.concat([test_usrid_merge_evt3_data, test_pre_usrid_merge_evt3_data],axis=0)

    test_df = pd.merge(test_agg_data, test_both_usrid_mergr_evt3_data, how='left', on='USRID')
    test_df.to_csv(test_temp,index=0)

    train_df.pop('USRID')
    X = train_df.as_matrix()

    train_flg_data.pop('USRID')
    Y = train_flg_data.as_matrix()

    test_df.pop('USRID')
    x_test = test_df.as_matrix()

    # 训练模型
    # lgr = LogisticRegression()
    # lgr.fit(X, Y)

    # y_pre_proba = lgr.predict_proba(x_test)
    
    xgb_model = xgboost.XGBClassifier(silent=1, max_depth=10, n_estimators=1000, learning_rate=0.05)
    xgb_model.fit(X, Y)
    y_pre_proba = xgb_model.predict_proba(x_test)

    # 取出训练到概率并存入文件
    pre_proba_to_csv(y_pre_proba[:, 1:])
    