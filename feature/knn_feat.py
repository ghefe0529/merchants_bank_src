# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 16:16:02
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 用knn填充数据

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

common_path = r'~/Documents/Study/Python/merchants_bank/'

train_log_evt3_usrid_path = common_path + r'/data/feature/train_usrid_evt3.csv'

test_agg_path = common_path + r'data/corpus/output/test_agg.csv'

if __name__ == '__main__':
    model = KNeighborsClassifier(n_neighbors=5)
    train_df = pd.read_csv(train_log_evt3_usrid_path)
    test_df = pd.read_csv()
    X = train_df[:1]
    Y = train_df[1:]
    model.fit(X, Y)
