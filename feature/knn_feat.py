# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 16:16:02
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 用knn填充数据

import numpy as np
import pandas as pd
# from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.neighbors import KNeighborsRegressor

common_path = r'~/Documents/Study/Python/merchants_bank'
# X
train_agg_with_log_path = common_path + r'/data/feature/train_agg_with_log.csv'
test_agg_with_log_path = common_path + r'/data/feature/test_agg_with_log.csv'

# Y
train_usrid_merge_evt3_path = common_path + r'/data/feature/train_usrid_merge_evt3.csv'
test_usrid_merge_evt3_path = common_path + r'/data/feature/test_usrid_merge_evt3.csv'
# x_test
train_agg_without_log_path = common_path + r'/data/feature/train_agg_without_log.csv'
test_agg_without_log_path = common_path + r'/data/feature/test_agg_without_log.csv'

# y_pre
train_pre_usrid_merge_evt3_withlog_path = common_path + r'/data/feature/train_pre_usrid_merge_evt3.csv'
test_pre_usrid_merge_evt3_withlog_path = common_path + r'/data/feature/test_pre_usrid_merge_evt3.csv'

def kNN_pre_merge_evt3(agg_with_log_path, usrid_merge_evt3_path, agg_without_log_path, pre_usrid_merge_evt3_withlog_path):
    X_0 = pd.read_csv(agg_with_log_path[0])
    X_1 = pd.read_csv(agg_with_log_path[1])
    X = pd.concat([X_0,X_1],axis=0)
    X.pop('USRID')

    Y_0 = pd.read_csv(usrid_merge_evt3_path[0])
    Y_1 = pd.read_csv(usrid_merge_evt3_path[1])
    Y = pd.concat([Y_0,Y_1],axis=0)
    Y.pop('USRID')

    x_test_0 = pd.read_csv(agg_without_log_path[0])
    train_n = x_test_0.shape[0]
    x_test_1 = pd.read_csv(agg_without_log_path[1])
    x_test = pd.concat([x_test_0,x_test_1],axis=0)

    y_usrid = x_test['USRID']
    # 初始化index
    y_usrid = y_usrid.reset_index(drop=True) 
    y_usrid = pd.DataFrame(y_usrid)
    # print('y[:10]',y_usrid[:10])
    
    x_test.pop('USRID')
    print('X shape', X.as_matrix().shape)
    print('Y shape', Y.as_matrix().shape)
    # print('Y[0]', type(Y.as_matrix().ravel()[0]))
    print('x_test shape', x_test.as_matrix().shape)
    print('model is begin')
    model = KNeighborsRegressor()
    model.fit(X.as_matrix(), Y.as_matrix().ravel())
    y_pre = model.predict(x_test.as_matrix())
    print('model is end')
    y_pre = pd.DataFrame(y_pre, columns=['y_pre'])

    print('y_pre shape ', y_pre.shape)
    # print('y_pre[:10]', y_pre[:10])
    print('y_pre type', type(y_pre))
    print('y_pre shape',y_pre[51120:])
    print('y_pre columns', y_pre.columns)
    print('y_pre index', y_pre.index)

    print('y_usrid shape',y_usrid.shape)
    print('y_usrid type', type(y_usrid))
    print('y_usrid shape',y_usrid[51120:])
    print('y_usrid columns', y_usrid.columns)
    print('y_usrid index', y_usrid.index)
    

    y_usrid_pre = pd.concat([y_usrid, y_pre], axis=1, ignore_index=True)
    print('writing')
    # print(pre_usrid_merge_evt3_withlog_path[0])
    # print(pre_usrid_merge_evt3_withlog_path[1])
    y_usrid_pre[:train_n].to_csv(pre_usrid_merge_evt3_withlog_path[0], index=0)
    y_usrid_pre[train_n:].to_csv(pre_usrid_merge_evt3_withlog_path[1], index=0)
    
if __name__ == '__main__':
    kNN_pre_merge_evt3([train_agg_with_log_path, test_agg_with_log_path], 
                        [train_usrid_merge_evt3_path, test_usrid_merge_evt3_path], 
                        [train_agg_without_log_path, test_agg_without_log_path], 
                        [train_pre_usrid_merge_evt3_withlog_path, test_pre_usrid_merge_evt3_withlog_path])
