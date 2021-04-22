import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest
import lightgbm as lgb
import csv
import re

# 创建列名
column_name = []
for i in range(2600):
    name = "attribute" + str(i+1)
    column_name.append(name)
print(column_name)

attr_data = []
with open("data_list.txt","r") as data_list:
    for line in data_list.readlines():
        txt_name = "./dataset/first_train_data/first_train_data/" + line.split("\n")[0] + ".txt"
        print(txt_name)
        with open(txt_name, "r") as f:
            attributes = f.readline()
            split_attr = re.split(',', attributes)
            # print(split_attr)
            # split_attr = np.array(split_attr).reshape(1, 2600)
            # print(split_attr)
            # print(split_attr)
            attr_data.append(split_attr)
            f.close()
        # print(attr_data)

attr_csv = pd.DataFrame(columns=column_name, data=attr_data)
attr_csv.to_csv('./attribute.csv', encoding='gbk')


#  as_matrix 将dataframe数据格式转化成数组






