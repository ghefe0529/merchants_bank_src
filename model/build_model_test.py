# -*- coding: utf-8 -*-
# @Date    : 2018-06-17 16:26:14
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 简单模型训练训练集

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# common_path = r'~/Documents/Study/Python/merchants_bank/'
common_path = r'~/Documents/merchants_bank'
# train
# agg 
train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'
# feat
train_one_two_path = common_path + r'/data/feature/1.2_train_usrid_merge_evt.csv'
train_two_path = common_path + r'/data/feature/2_train_log_count.csv'
train_three_path = common_path + r'/data/feature/3_train_time.csv'
train_four_path = common_path + r'/data/feature/4_train_tfidf.csv'
train_five_path = common_path + r'/data/feature/5_train_tfidf_stack.csv'
train_six_path = common_path + r'/data/feature/6_train_new_agg.csv'

# flg
train_flg_path = common_path + r'/data/corpus/output/train_flg.csv'

# test
# agg 
test_agg_path = common_path + r'/data/corpus/output/test_agg.csv'
# feat
test_one_two_path = common_path + r'/data/feature/1.2_test_usrid_merge_evt.csv'
test_two_path = common_path + r'/data/feature/2_test_log_count.csv'
test_three_path = common_path + r'/data/feature/3_test_time.csv'
test_four_path = common_path + r'/data/feature/4_test_tfidf.csv'
test_five_path = common_path + r'/data/feature/5_test_tfidf_stack.csv'
test_six_path = common_path + r'/data/feature/6_test_new_agg.csv'

# other feat
all_lasttime_feature_path = common_path + r'/data/all_lasttime_feature.csv'
all_maxclick_feature_path = common_path + r'/data/all_maxclick_feature.csv'
all_train_path = common_path + r'/data/all_train.csv'
all_test_path = common_path + r'/data/all_test.csv'
final_time_path = common_path + r'/data/Final_time.csv'
final_test_time_path = common_path + r'/data/Final_Test_time.csv'
all_test_path = common_path + r'/data/all_test.csv'
all_train_path = common_path + r'/data/all_train.csv'

# result
result_path = common_path + r'/data/result1.csv'
# 暂存train和test
train_temp_path = common_path + r'/data/feature/train_temp.csv'
test_temp_path = common_path + r'/data/feature/test_temp.csv'

# 将特征拼接起来
def concat_data(data, concat_data_path, isfillna=True):
    concat_data = pd.read_csv(concat_data_path)
    data = pd.merge(data, concat_data, how='left', on='USRID')
    if isfillna:
        data.fillna(0)
    return data

def select_feature(train_df, train_flg_df, n=100, mode='first'):
    print('train_df shape ', train_df.shape)
    xgb_model = XGBClassifier(booster = 'gbtree',
              objective = 'binary:logistic',
              eta = 0.02,
              max_depth = 4,  # 4 3
              colsample_bytree = 0.8,#0.8
              subsample = 0.7,
              min_child_weight = 9,  # 2 3
              silent=1)
    xgb_model.fit(train_df.values,train_flg_df.values.ravel())
    columns = train_df.columns
    imp = xgb_model.feature_importances_
    columns_imp = {}
    for i in range(len(columns)):
        columns_imp[columns[i]] = imp[i]
    columns_imp = sorted(columns_imp.items(), key= lambda x: x[1], reverse=True)
    print(columns_imp)
    columns_imp = columns_imp[0:n]
    if mode == 'first':
        feature_list = [ j[0] for j in columns_imp ]
    else:
        feature_list = [ j[0] for j in columns_imp if j[1] != 0 ]
    print('feature_list len ', len(feature_list))
    return feature_list


# 将测试结果保存至文件
def pre_proba_to_csv(pre_proba):
    
    # 获取测试集的USRID
    test_agg_usrid_data = pd.read_csv(test_agg_path)['USRID']
    # 将概率转化为DataFrame
    result_data_pre = pd.DataFrame(pre_proba, columns=['RST'])
    # 合并概率和USRID为DataFrame
    result_data = pd.concat([test_agg_usrid_data, result_data_pre],axis=1)

    # 存储结果
    result_data.to_csv(result_path, index=0, sep='\t')


if __name__ == '__main__':
    # 读取训练集
    train_agg_data = pd.read_csv(train_agg_path)
    train_df = concat_data(train_agg_data, train_one_two_path)
    train_df = concat_data(train_df, train_two_path)
    train_df = concat_data(train_df, train_three_path)
    # train_df = concat_data(train_df, train_four_path)
    train_df = concat_data(train_df, train_five_path)
    train_df = concat_data(train_df, train_six_path)
    # 读取测试集
    test_agg_data = pd.read_csv(test_agg_path)
    test_df = concat_data(test_agg_data, test_one_two_path)
    test_df = concat_data(test_df, test_two_path)
    test_df = concat_data(test_df, test_three_path)
    # test_df = concat_data(test_df, test_four_path)
    test_df = concat_data(test_df, test_five_path)
    test_df = concat_data(test_df, test_six_path)

    # 读取flg
    train_flg_data = pd.read_csv(train_flg_path)

    # 拼接other feat
    # 读取其他特征
    all_lasttime_df = pd.read_csv(all_lasttime_feature_path)
    # 计算最后一次点击时间的平均值
    fill_time = sum(list(all_lasttime_df['LASTTIME']))/len(list(all_lasttime_df['LASTTIME']))
    all_maxclick_df = pd.read_csv(all_maxclick_feature_path)

    final_time_df = pd.read_csv(final_time_path)
    final_test_time_df = pd.read_csv(final_test_time_path)
    # 将其他特征与train合并
    train_df = pd.merge(train_df, all_lasttime_df, how='left',on='USRID')
    # train_df = train_df.fillna(fill_time)
    train_df = pd.merge(train_df, all_maxclick_df, how='left',on='USRID')
    train_df = pd.merge(train_df, final_time_df, how='left', on='USRID')
    # 将其他特征与test合并
    test_df = pd.merge(test_df, all_lasttime_df, how='left',on='USRID')
    # test_df = test_df.fillna(fill_time)
    test_df = pd.merge(test_df, all_maxclick_df, how='left',on='USRID')
    test_df = pd.merge(test_df, final_test_time_df, how='left', on='USRID')

    # 添加老师的特征
    '''
    all_train_df = pd.read_csv(all_train_path)
    all_test_df = pd.read_csv(all_test_path)

    train_agg_data.pop('USRID')
    test_agg_data.pop('USRID')

    all_train_df.pop('recenttime')
    all_train_df.pop('FLAG')
    all_test_df.pop('recenttime')

    all_train_df.drop(train_agg_data.columns, axis=1, inplace=True)
    all_test_df.drop(test_agg_data.columns, axis=1, inplace=True)

    train_df = pd.merge(train_df,all_train_df,how='left',on='USRID')
    test_df = pd.merge(test_df,all_test_df,how='left',on='USRID')
    '''

    # 保存合并后的train和test
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    train_df.to_csv(train_temp_path, index=0)
    test_df.to_csv(test_temp_path, index=0)

    # 删除USRID
    train_df.pop('USRID')
    test_df.pop('USRID')
    train_flg_data.pop('USRID')

    # imp_feat = select_feature(train_df, train_flg_data, 50)
    imp_feat = select_feature(train_df, train_flg_data, mode='second')

    X = train_df[imp_feat].values
    Y = train_flg_data.values.ravel()
    X_test = test_df[imp_feat].values

    # 用拆分训练集来预测结果
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    # 存储5次测试的结果
    test_auc_list = []
    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # 训练模型
        print('test model is begin')
        xgb_model = XGBClassifier(booster = 'gbtree',
                objective = 'binary:logistic',
                eta = 0.02,
                max_depth = 4,  # 4 3
                colsample_bytree = 0.8,#0.8
                subsample = 0.7,
                min_child_weight = 9,  # 2 3
                silent=1)

        # xgb_model = XGBClassifier(learning_rate=0.01,max_depth=4,n_estimators=800,n_jobs=4)
        
        xgb_model.fit(x_train, y_train)
        y_pre_proba = xgb_model.predict_proba(x_test)[:, 1:]

        print('test model is end')
        test_auc = roc_auc_score(y_test, y_pre_proba)
        print('test auc ', test_auc)
        test_auc_list.append(test_auc)
    print(sum(test_auc_list)/5)
    



    # 训练真正的模型
    print('model is begin')

    xgb_model = XGBClassifier(booster = 'gbtree',
              objective = 'binary:logistic',
              eta = 0.02,
              max_depth = 4,  # 4 3
              colsample_bytree = 0.8,#0.8
              subsample = 0.7,
              min_child_weight = 9,  # 2 3
              silent=1)

    # xgb_model = XGBClassifier(learning_rate=0.01,max_depth=4,n_estimators=800,n_jobs=4)
    
    xgb_model.fit(X, Y)
    y_pre_proba = xgb_model.predict_proba(X_test)[:, 1:]

    # xgb_model = XGBClassifier(silent=1, max_depth=10, n_estimators=1000, learning_rate=0.05)
    # xgb_model.fit(X, Y)
    # y_pre_proba = xgb_model.predict_proba(x_test)
    print('model is end')

    # 取出训练到概率并存入文件
    pre_proba_to_csv(y_pre_proba)