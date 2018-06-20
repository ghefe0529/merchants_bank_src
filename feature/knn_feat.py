# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 16:16:02
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 用knn填充数据

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

common_path = r'~/Documents/Study/Python/merchants_bank/'

train_usrid_evt3_agg_path = common_path + r'/data/feature/train_usrid_evt3_agg.csv'

train_agg_without_log_path = common_path + r'/data/feature/train_agg_without_log.csv'

test_usrid_evt3_agg_path = common_path + r'data/corpus/output/test_usrid_ect3_agg.csv'

if __name__ == '__main__':
    model = KNeighborsClassifier(n_neighbors=5)
    train_df = pd.read_csv(train_usrid_evt3_agg_path)
    train_df_columns = train_df.columns
    print(list(train_df_columns))
    print(type(list(train_df_columns)))
    # X = train_df
    # Y =train_df
    # model.fit(X, Y)
