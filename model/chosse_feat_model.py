# -*- coding: utf-8 -*-
# @Date    : 2018-06-25 09:36:19
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe :  用lasso做特征选择再训练


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,Lasso,LinearRegression,Ridge
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# common_path = r'~/Documents/Study/Python/merchants_bank/'
common_path = r'~/Documents/merchants_bank'

# temp
train_temp = common_path + r'/data/feature/trian_temp.csv'
other_train_path = common_path + r'/data/all_train.csv'
# flg
train_flg_path = common_path + r'/data/corpus/output/train_flg.csv'

# temp
test_temp = common_path + r'/data/feature/test_temp.csv'
other_test_path = common_path + r'/data/all_test.csv'
# flg

# 结果存放文件
result_path = common_path + r'/data/feature/result_test7.csv'


def select_coef_by_model(coef, data, min=0.0000001):
    del_weight = []
    for ele,index in zip(coef, range(len(coef))):
        if np.abs(ele) == 0:
            del_weight.append(index)
    data = np.delete(data, del_weight, axis=1)
    return data,del_weight

def pre_proba_to_csv(pre_proba):
    
    # 获取测试集的USRID
    test_agg_usrid_data = pd.read_csv(test_temp)['USRID']
    # 将概率转化为DataFrame
    result_data_pre = pd.DataFrame(pre_proba, columns=['RST'])
    # 合并概率和USRID为DataFrame
    result_data = pd.concat([test_agg_usrid_data, result_data_pre],axis=1)

    # print(result_data[:10])
    # 存储结果
    result_data.to_csv(result_path, index=0, sep='\t')

if __name__ == '__main__':
    train_df = pd.read_csv(train_temp)

    other_train_df = pd.read_csv(other_train_path)
    # print('other_train columns', other_train_df.columns)
    v_list = ['V%s' % x for x in range(1,31) ]
    v_list.append('FLAG')
    # print(v_list)
    other_train_df = other_train_df.drop(v_list, axis=1)

    train_df = pd.merge(train_df, other_train_df, on='USRID',how='left')

    flg_df = pd.read_csv(train_flg_path)
    
    test_df = pd.read_csv(test_temp)

    other_test_df = pd.read_csv(other_test_path)
    v_list.remove('FLAG')
    other_test_df = other_test_df.drop(v_list, axis=1)

    test_df = pd.merge(test_df, other_test_df, on='USRID',how='left')

    # print('test columns', test_df.columns)

    # print('other_test columns', other_test_df.columns)

    print('train_df shape',train_df.shape)
    print('test_df shape', test_df.shape)

    # print('train columns', list(train_df.columns))
    # print('test columns', list(test_df.columns))

    train_df.pop('USRID')
    train_df.pop('recenttime')

    flg_df.pop('USRID')
    test_df.pop('recenttime')
    test_df.pop('USRID')
    
    
    print('train_df shape',train_df.shape)
    print('flg_df shape',flg_df.shape)
    print('test_df shape', test_df.shape)

    # '''
    print('------lasso begin-----')
    # lasso_model = Lasso(alpha=0.000001)
    # lasso_model = Ridge(alpha=0.001)
    lasso_model = RandomForestRegressor(n_estimators=200, max_features=6)

    lasso_model.fit(train_df.values, flg_df.values.ravel())
    print('------lasso end-----')

    # print('coef len ',len(lasso_model.coef_))
    print('coef len ',len(lasso_model.feature_importances_))

    # print('coef ', lasso_model.coef_)
    # train_new, dorp_columns = select_coef_by_model(lasso_model.coef_, train_df.values)
    train_new, dorp_columns = select_coef_by_model(lasso_model.feature_importances_, train_df.values)
    test_new = np.delete(test_df.values, dorp_columns, axis=1) 
    print('train new :', train_new.shape)
    print('test new :', test_new.shape)
    x_train, x_test, y_train, y_test = train_test_split(train_new, flg_df,test_size=0.2)
    # '''
    # x_train, x_test, y_train, y_test = train_test_split(train_df, flg_df,test_size=0.2)
    xgb_model = XGBClassifier(booster = 'gbtree',
                objective = 'binary:logistic',
                eta = 0.02,
                max_depth = 4,  # 4 3
                colsample_bytree = 0.8,#0.8
                subsample = 0.7,
                min_child_weight = 9,  # 2 3
                n_jobs = 4,
                silent = 1)

    xgb_model.fit(x_train, y_train.values.ravel())


    y_pre_proba = xgb_model.predict_proba(x_test)[:, 1:]

    test_auc = metrics.roc_auc_score(y_test,y_pre_proba)
    
    print('test_auc is ',test_auc)    

    model = XGBClassifier(booster = 'gbtree',
                objective = 'binary:logistic',
                eta = 0.02,
                max_depth = 4,  # 4 3
                colsample_bytree = 0.8,#0.8
                subsample = 0.7,
                min_child_weight = 9,  # 2 3
                n_jobs = 2,
                silent = 1)

    model.fit(train_new, flg_df.values.ravel())
    # model.fit(train_df.values, flg_df.values.ravel())


    
    y_pre_proba = model.predict_proba(test_new)[:, 1:]
    # y_pre_proba = model.predict_proba(test_df.values)[:, 1:]

    pre_proba_to_csv(y_pre_proba)
