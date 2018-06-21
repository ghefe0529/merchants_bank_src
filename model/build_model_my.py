# -*- coding: utf-8 -*-
# @Date    : 2018-06-17 16:26:14
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 简单模型训练训练集

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import xgboost
from sklearn import metrics

common_path = r'~/Documents/Study/Python/merchants_bank/'
# train 
train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'
# merge_evt3
train_usrid_merge_evt3_path = common_path + r'/data/feature/train_usrid_merge_evt3.csv'
train_pre_usrid_merge_evt3_path = common_path + r'/data/feature/train_pre_usrid_merge_evt3.csv'
# log_count
train_log_count_path = common_path + r'/data/feature/train_log_count.csv'
# time_feat
train_time_path = common_path + r'data/feature/train_time.csv'

# train_log_path = r'/data/corpus/train_log.csv'
train_flg_path = common_path + r'data/corpus/output/train_flg.csv'

# temp
train_temp = common_path + r'/data/feature/trian_temp.csv'

# test
test_agg_path = common_path + r'data/corpus/output/test_agg.csv'
# merge_evt3
test_usrid_merge_evt3_path = common_path + r'/data/feature/test_usrid_merge_evt3.csv'
test_pre_usrid_merge_evt3_path = common_path + r'/data/feature/test_pre_usrid_merge_evt3.csv'
# log_count
test_log_count_path = common_path + r'/data/feature/test_log_count.csv'
# time_feat
test_time_path = common_path + r'data/feature/test_time.csv'

# temp
test_temp = common_path + r'/data/feature/test_temp.csv'

result_path = common_path + r'data/corpus/output/test_result1.csv'

def get_median(tmp_list):
    tmp_list.sort()
    size = len(tmp_list)
    if size % 2 == 0:
        return (tmp_list[size//2]+tmp_list[size//2-1])/2
    else:
        return tmp_list[(size-1)//2]

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
    '''
    # 读取训练集 加上了evt3
    train_agg_data = pd.read_csv(train_agg_path)
    train_usrid_merge_evt3_data = pd.read_csv(train_usrid_merge_evt3_path)
    train_pre_usrid_merge_evt3_data = pd.read_csv(train_pre_usrid_merge_evt3_path)
    train_both_usrid_mergr_evt3_data = pd.concat([train_usrid_merge_evt3_data, train_pre_usrid_merge_evt3_data],axis=0)

    train_df = pd.merge(train_agg_data,train_both_usrid_mergr_evt3_data, how='left', on='USRID')
    train_df.to_csv(train_temp,index=0)

    # Y
    train_flg_data = pd.read_csv(train_flg_path)

    train_df.pop('USRID')
    X = train_df.as_matrix()

    train_flg_data.pop('USRID')
    Y = train_flg_data.as_matrix()

    # 训练模型

    # 逻辑回归
    x_train, x_test, y_train, y_test = train_test_split(X, Y.ravel(), test_size=0.2, random_state=0)
    lgr = LogisticRegression()
    lgr.fit(x_train, y_train)
    y_pre_proba = lgr.predict_proba(x_test)[:, 1:]
    
    # xgboost
    # xgb_model = xgboost.XGBClassifier(silent=1, max_depth=10, n_estimators=1000, learning_rate=0.05)
    # xgb_model.fit(x_train, y_train)
    # y_pre_proba = xgb_model.predict_proba(x_test)[:, 1:]

    # 高斯朴素贝叶斯
    # gss = GaussianNB()
    # gss.fit(x_train, y_train)
    # y_pre_proba = gss.predict_proba(x_test)[:, 1:]

    test_auc = metrics.roc_auc_score(y_test,y_pre_proba)
    
    print(test_auc)    

    # 取出训练到概率并存入文件
    # pre_proba_to_csv(y_pre_proba[:, 1:])
    '''
    # 读取训练集 加上count
    train_agg_data = pd.read_csv(train_agg_path)
    # 添加count 没有的填充平均数
    train_log_count_data = pd.read_csv(train_log_count_path)

    train_df = pd.merge(train_agg_data, train_log_count_data, how='left', on='USRID')
    
    print(train_df['LOG_COUNT'][1])
    # median_count = get_median(list(train_df['LOG_COUNT']))
    train_dt_tmpe = train_df.fillna(0)

    avg_count = sum(list(train_dt_tmpe['LOG_COUNT']))/len(list(train_dt_tmpe['LOG_COUNT']))
    # print('median is ', median_count)
    train_df = train_df.fillna(avg_count)


    train_df.to_csv(train_temp,index=0)

    # Y
    train_flg_data = pd.read_csv(train_flg_path)

    # turn X
    train_df.pop('USRID')
    X = train_df.as_matrix()
    
    # turn Y
    train_flg_data.pop('USRID')
    Y = train_flg_data.as_matrix()

    # 训练模型

    # 逻辑回归
    x_train, x_test, y_train, y_test = train_test_split(X, Y.ravel(), test_size=0.2, random_state=0)
    lgr = LogisticRegression()
    lgr.fit(x_train, y_train)
    y_pre_proba = lgr.predict_proba(x_test)[:, 1:]
    
    # xgboost
    # xgb_model = xgboost.XGBClassifier(silent=1, max_depth=10, n_estimators=1000, learning_rate=0.05)
    # xgb_model.fit(x_train, y_train)
    # y_pre_proba = xgb_model.predict_proba(x_test)[:, 1:]

    # 高斯朴素贝叶斯
    # gss = GaussianNB()
    # gss.fit(x_train, y_train)
    # y_pre_proba = gss.predict_proba(x_test)[:, 1:]

    test_auc = metrics.roc_auc_score(y_test,y_pre_proba)
    
    print(test_auc)    

    # 取出训练到概率并存入文件
    # pre_proba_to_csv(y_pre_proba[:, 1:])