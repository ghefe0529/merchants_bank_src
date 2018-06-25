# -*- coding: utf-8 -*-
# @Date    : 2018-06-17 16:26:14
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 简单模型训练训练集

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import xgboost
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# common_path = r'~/Documents/Study/Python/merchants_bank/'
common_path = r'~/Documents/merchants_bank'
# train 
train_agg_path = common_path + r'/data/corpus/output/train_agg.csv'
# train_new_agg_path = common_path + r'/data/feature/train_new_agg.csv'
# merge_evt3
train_usrid_merge_evt3_path = common_path + r'/data/feature/train_usrid_merge_evt3.csv'
train_pre_usrid_merge_evt3_path = common_path + r'/data/feature/train_pre_usrid_merge_evt3.csv'
# log_count
train_log_count_path = common_path + r'/data/feature/train_log_count.csv'
train_pre_log_count_path = common_path + r'/data/feature/train_pre_log_count.csv'

# time_feat
train_time_path = common_path + r'/data/feature/train_time.csv'
train_pre_time_path = common_path + r'/data/feature/train_pre_time.csv'
# tfidf
train_tfidf_path = common_path + r'/data/feature/train_tfidf.csv'
# time two
train_time_two_path = common_path + r'/data/Final_time.csv'
# last time
train_last_time_path = common_path + r'/data/all_lasttime_feature.csv'
# max click
train_max_click_path = common_path +r'/data/all_lasttime_feature.csv'


# train_log_path = r'/data/corpus/train_log.csv'
train_flg_path = common_path + r'/data/corpus/output/train_flg.csv'

# temp
train_temp = common_path + r'/data/feature/trian_temp.csv'

# test
test_agg_path = common_path + r'data/corpus/output/test_agg.csv'
# test_new_agg_path = common_path + r'/data/feature/test_new_agg.csv'
# merge_evt3
test_usrid_merge_evt3_path = common_path + r'/data/feature/test_usrid_merge_evt3.csv'
test_pre_usrid_merge_evt3_path = common_path + r'/data/feature/test_pre_usrid_merge_evt3.csv'
# log_count
test_log_count_path = common_path + r'/data/feature/test_log_count.csv'
test_pre_log_count_path = common_path + r'/data/feature/test_pre_log_count.csv'
# time_feat
test_time_path = common_path + r'data/feature/test_time.csv'
test_pre_time_path = common_path + r'data/feature/test_pre_time.csv'
# tfidf
test_tfidf_path = common_path + r'/data/feature/test_tfidf.csv'

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
    
    # log_count 
    train_log_count_data = pd.read_csv(train_log_count_path)
    train_pre_log_count_data = pd.read_csv(train_pre_log_count_path)
    train_both_log_count_data = pd.concat([train_log_count_data, train_pre_log_count_data], axis=0)

    train_df = pd.merge(train_df, train_both_log_count_data, how='left', on='USRID')

    # time
    train_time_data = pd.read_csv(train_time_path)
    train_pre_time_data = pd.read_csv(train_pre_time_path)
    train_both_time_data = pd.concat([train_time_data, train_pre_time_data], axis=0)

    train_df = pd.merge(train_df, train_both_time_data, how='left', on='USRID')

    # 训练集写入文件
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

    test_auc = metrics.roc_auc_score(y_test, y_pre_proba)
    
    print(test_auc)    

    # 取出训练到概率并存入文件
    # pre_proba_to_csv(y_pre_proba[:, 1:])
    '''
    # -----------------------------------------------------------
    # 读取训练集
    train_agg_data = pd.read_csv(train_agg_path)
    
    # -----------------------------------------------------------
    # 读取evt3
    train_usrid_merge_evt3_data = pd.read_csv(train_usrid_merge_evt3_path)
    # train_pre_usrid_merge_evt3_data = pd.read_csv(train_pre_usrid_merge_evt3_path)
    # train_both_usrid_merge_evt3_data = pd.concat([train_usrid_merge_evt3_data, train_pre_usrid_merge_evt3_data],axis=0)

    train_df = pd.merge(train_agg_data,train_usrid_merge_evt3_data, how='left', on='USRID')
    # train_df = pd.merge(train_agg_data,train_both_usrid_merge_evt3_data, ow='left', on='USRID')
    train_dt_tmpe = train_df.fillna(0)


    # -----------------------------------------------------------
    # 添加count 没有的填充平均数
    # 取出点击次数
    train_log_count_data = pd.read_csv(train_log_count_path)

    # agg + count
    train_df = pd.merge(train_df, train_log_count_data, how='left', on='USRID')
    
    # print(train_df['LOG_COUNT'][1])
    # median_count = get_median(list(train_df['LOG_COUNT']))

    # 没有count的填充平均数
    train_dt_tmpe = train_df.fillna(0)

    avg_count = sum(list(train_dt_tmpe['LOG_COUNT']))/len(list(train_dt_tmpe['LOG_COUNT']))
    # print('median is ', median_count)
    train_df = train_df.fillna(avg_count)
    
    # -----------------------------------------------------------
    # 添加单位时间点击的平均数
    # 取出单位时间点击的平均数
    train_time_data =  pd.read_csv(train_time_path)

    train_df = pd.merge(train_df, train_time_data, how='left', on='USRID')
    # 没有time 的填充0
    train_df = train_df.fillna(0)


    # -----------------------------------------------------------
    # 添加tfidf
    train_tfidf_data = pd.read_csv(train_tfidf_path)

    train_df = pd.merge(train_df, train_tfidf_data, how='left',left_on='USRID',right_on='0')
    train_df = train_df.fillna(0)

    # -----------------------------------------------------------
    # 添加点击天数/点击总数的特征
    train_time_two_data = pd.read_csv(train_time_two_path)
    train_df = pd.merge(train_df, train_time_two_data, how='left', on='USRID')
    train_df = train_df.fillna(0)


    # -----------------------------------------------------------
    # 最后一次点击次数
    train_last_time_data = pd.read_csv(train_last_time_path)
    train_df = pd.merge(train_df, train_last_time_data, how='left', on='USRID')
    train_df = train_df.fillna(0)



    # -----------------------------------------------------------
    # 最高点击频率
    train_max_click_data = pd.read_csv(train_max_click_path)
    train_df = pd.merge(train_df, train_max_click_data, how='left', on='USRID')
    train_df = train_df.fillna(0)



    train_df.to_csv(train_temp,index=0)

    # Y
    train_flg_data = pd.read_csv(train_flg_path)

    # turn X
    train_df.pop('USRID')
    X = train_df.values
    
    # turn Y
    train_flg_data.pop('USRID')
    Y = train_flg_data.values

    # 训练模型

    # 逻辑回归
    x_train, x_test, y_train, y_test = train_test_split(X, Y.ravel(), test_size=0.2, random_state=0)
    # lgr = LogisticRegression()
    # lgr.fit(x_train, y_train)
    # y_pre_proba = lgr.predict_proba(x_test)[:, 1:]
    
    # xgboost
    # silent过程是否输出（0：输出）
    # nthread线程数
    # learning_rate 通过减少每一步的权重，可以提高模型的鲁棒性。典型值为0.01-0.2。
    # min_child_weight 决定最小叶子节点样本权重和。但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
    # max_depth 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。需要使用CV函数来进行调优。典型值：3-10
    # max_leaf_nodes 树上最大的节点或叶子的数量。如果定义了这个参数，GBM会忽略max_depth参数。
    # gamma 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。Gamma指定了节点分裂所需的最小损失函数下降值。
    # subsample 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。典型值：0.5-1
    # reg_lambda 权重的L2正则化项 (和Ridge regression类似)。
    # reg_alpha 权重的L1正则化项 (和Lasso regression类似)。
    # scale_pos_weight 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
    # objective 
    #           binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
    #             multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。 
    #                     在这种情况下，你还需要多设一个参数：num_class(类别数目)。
    #             multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
    # parameters = {
    #     'silent':[1],
    #     'objective':['binary:logistic','reg:linear'],
    #     'max_depth' :[4,6,8,10],
    #     'learning_rate':[0.1,0.01,0.001],
    #     'n_estimators':[100,200,500,800,1000],
    #     'reg_lambda':[0.5,0.8,1]
    # }
    # xgb_model = xgboost.XGBClassifier()
    # clf = GridSearchCV(xgb_model, parameters, scoring='roc_auc', n_jobs=4)
    # clf.fit(x_train,y_train)
    # # xgb_model.fit(x_train, y_train)
    # y_pre_proba = xgb_model.predict_proba(x_test)[:, 1:]

    # xgb_model = xgboost.XGBClassifier(learning_rate=0.01,max_depth=4,n_estimators=800,n_jobs=4)

    xgb_model = xgboost.XGBClassifier(booster = 'gbtree',
              objective = 'binary:logistic',
              eta = 0.02,
              max_depth = 4,  # 4 3
              colsample_bytree = 0.8,#0.8
              subsample = 0.7,
              min_child_weight = 9,  # 2 3
              n_jobs = 4,
              silent = 1)

    xgb_model.fit(x_train, y_train)
    y_pre_proba = xgb_model.predict_proba(x_test)[:, 1:]

    # 高斯朴素贝叶斯
    # gss = GaussianNB()
    # gss.fit(x_train, y_train)
    # y_pre_proba = gss.predict_proba(x_test)[:, 1:]

    # print('The parameters of the best model are: ') 
    # print(clf.best_params_) # 打印出最合适的模型参数 
    test_auc = metrics.roc_auc_score(y_test,y_pre_proba)
    
    print(test_auc)    

    # 取出训练到概率并存入文件
    # pre_proba_to_csv(y_pre_proba[:, 1:])
    # '''