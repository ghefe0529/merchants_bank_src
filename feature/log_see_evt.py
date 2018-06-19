# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 10:36:46
# @Author  : GEFE (gh_efe@163.com)
# @Version : 1.0.0
# @Describe : 查看EVT的情况

import numpy as np
import pandas as pd

common_path = r'~/Documents/Study/Python/merchants_bank/'

train_log_corpus_path = common_path + r'/data/feature/train_log.csv'

train_log_evt_path = common_path + r'/data/feature/train_log_evt.csv'

if __name__ == '__main__':
    train_log_corpus_data = pd.read_csv(train_log_corpus_path)['EVT_LBL']
    train_log_corpus_data = set(train_log_corpus_data.as_matrix())
    train_log_corpus_data = [ x+'' for x in train_log_corpus_data ]
    train_log_corpus_data = [ x.split('-') for x in train_log_corpus_data]
    # train_log_corpus_data = pd.DataFrame(train_log_corpus_data)
    # print(train_log_corpus_data)
    sum = len(train_log_corpus_data)
    for ele1 in train_log_corpus_data:
        count = 0
        for ele2 in train_log_corpus_data:
            if ele1 == ele2:
                count += 1
        if count != 1:
            print(ele1)
        print(count)
    print(sum)
    # train_log_corpus_data = train_log_corpus_data.sort_values(by=2, ascending=True)
    # train_log_corpus_data.to_csv(train_log_evt_path, index=0)
