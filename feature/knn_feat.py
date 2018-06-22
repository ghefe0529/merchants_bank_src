# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 16:16:02
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 用knn填充数据

import numpy as np
import pandas as pd
# from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso,LinearRegression

# common_path = r'~/Documents/Study/Python/merchants_bank'
common_path = r'~/Documents/merchants_bank'

# X
train_agg_with_log_path = common_path + r'/data/feature/train_agg_with_log.csv'
test_agg_with_log_path = common_path + r'/data/feature/test_agg_with_log.csv'

# Y
# merge_evt3
train_usrid_merge_evt3_path = common_path + r'/data/feature/train_usrid_merge_evt3.csv'
test_usrid_merge_evt3_path = common_path + r'/data/feature/test_usrid_merge_evt3.csv'
# log_count
train_log_count_path = common_path + r'/data/feature/train_log_count.csv'
test_log_count_path = common_path + r'/data/feature/test_log_count.csv'
# time
train_time_path = common_path + r'/data/feature/train_time.csv'
test_time_path = common_path + r'/data/feature/test_time.csv'

# x_test
train_agg_without_log_path = common_path + r'/data/feature/train_agg_without_log.csv'
test_agg_without_log_path = common_path + r'/data/feature/test_agg_without_log.csv'

# y_pre
# merge_evt3
train_pre_usrid_merge_evt3_withlog_path = common_path + r'/data/feature/train_pre_usrid_merge_evt3.csv'
test_pre_usrid_merge_evt3_withlog_path = common_path + r'/data/feature/test_pre_usrid_merge_evt3.csv'

# log_count
train_pre_log_count_withlog_path = common_path + r'/data/feature/train_pre_log_count.csv'
test_pre_log_count_withlog_path = common_path + r'/data/feature/test_pre_log_count.csv'

# time
train_pre_time_withlog_path = common_path + r'/data/feature/train_pre_time.csv'
test_pre_time_withlog_path = common_path + r'/data/feature/test_pre_time.csv'

def kNN_pre_without(X_path, Y_path, x_test_path, y_test_path, feat_name):
    # 读取有log的usrid的agg
    X_0 = pd.read_csv(X_path[0])
    X_1 = pd.read_csv(X_path[1])
    X = pd.concat([X_0,X_1],axis=0)
    X.pop('USRID')
    # 读取merge_evt3
    Y_0 = pd.read_csv(Y_path[0])
    Y_1 = pd.read_csv(Y_path[1])
    Y = pd.concat([Y_0,Y_1],axis=0)
    Y.pop('USRID')
    # 读取没有log的usrid的agg
    x_test_0 = pd.read_csv(x_test_path[0])
    train_n = x_test_0.shape[0]
    x_test_1 = pd.read_csv(x_test_path[1])
    # 将train和test合并
    x_test = pd.concat([x_test_0,x_test_1],axis=0)
    
    # 初始化index，因为是两个文件所有index不一样
    x_test = x_test.reset_index(drop=True) 
    x_test = pd.DataFrame(x_test)

    # 获取train和test内没有log的usrid的usrid列表
    y_usrid = x_test['USRID']
    # 初始化index，因为是两个文件所有index不一样
    y_usrid = y_usrid.reset_index(drop=True) 
    y_usrid = pd.DataFrame(y_usrid)
    
    x_test.pop('USRID')

    # print('X shape', X.as_matrix().shape)
    # print('Y shape', Y.as_matrix().shape)
    # print('Y[0]', type(Y.as_matrix().ravel()[0]))
    # print('x_test shape', x_test.as_matrix().shape)
    print('model is begin')

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X.as_matrix(), Y.as_matrix().ravel())
    y_pre = model.predict(x_test.as_matrix())
    print('model is end')
    y_pre = pd.DataFrame(y_pre, columns=['y_pre'])

    y_usrid_pre = pd.concat([y_usrid, y_pre], axis=1)
    y_usrid_pre.columns=['USRID', feat_name]
    print('writing')
    # print(pre_usrid_merge_evt3_withlog_path[0])
    # print(pre_usrid_merge_evt3_withlog_path[1])
    # y_usrid_pre = pd.DataFrame(y_usrid_pre, columns=['USRID','merge_evt3'])
    y_usrid_pre[:train_n].to_csv(y_test_path[0], index=0)
    y_usrid_pre[train_n:].to_csv(y_test_path[1], index=0)
    

if __name__ == '__main__':
    # '''
    # 预测evt3
    kNN_pre_without([train_agg_with_log_path, test_agg_with_log_path], 
                        [train_usrid_merge_evt3_path, test_usrid_merge_evt3_path], 
                        [train_agg_without_log_path, test_agg_without_log_path], 
                        [train_pre_usrid_merge_evt3_withlog_path, test_pre_usrid_merge_evt3_withlog_path],
                        'MERGE_EVT3')
    
    # 预测log count
    kNN_pre_without([train_agg_with_log_path, test_agg_with_log_path], 
                        [train_log_count_path, test_log_count_path],
                        [train_agg_without_log_path, test_agg_without_log_path], 
                        [train_pre_log_count_withlog_path, test_pre_log_count_withlog_path],
                        'LOG_COUNT')
    # '''
    # 预测time
    kNN_pre_without([train_agg_with_log_path, test_agg_with_log_path], 
                        [train_time_path, test_time_path],
                        [train_agg_without_log_path, test_agg_without_log_path], 
                        [train_pre_time_withlog_path, test_pre_time_withlog_path],
                        'avg_tim')
