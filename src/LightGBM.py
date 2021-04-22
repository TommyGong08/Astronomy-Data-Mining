import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from pandas.core.frame import DataFrame
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score
import lightgbm as lgb

'''
# 制作X_train
train_data_df = pd.read_csv("attribute.csv")
train_data_df.drop(['Unnamed: 0'], inplace=True, axis=1)
# print(train_data_df)
X_train = train_data_df.values
print(X_train.shape)

# 制作Y_train
y_train_df = pd.read_csv("label.csv")
y_train_df.loc[y_train_df['type'] == 'star'] = 0
y_train_df.loc[y_train_df['type'] == 'galaxy'] = 1
y_train_df.loc[y_train_df['type'] == 'qso'] = 2
y_train_df.loc[y_train_df['type'] == 'unknown'] = 3
y_train_df = y_train_df['type']
Y_train = y_train_df.values
print(Y_train)
'''

# 开始预测
# 将label.csv和attributes合并 之后划分验证集交叉验证
test_data = pd.read_csv("test_data.csv")
test_data.drop(['Unnamed: 0'], inplace=True, axis=1)
print(test_data)
test_data_np = test_data.values
print(test_data_np)

X_test_data = test_data_np[:, 0:2600]
Y_test_data = test_data_np[:, 2600]

# K-Fold方法划分数据集和验证集
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X_test_data):
    print('train_index', train_index, 'test_index', test_index)
    train_X, train_y = X_test_data[train_index], Y_test_data[train_index]
    test_X, test_y = X_test_data[test_index], Y_test_data[test_index]

# 构造lgb
lgb_train = lgb.Dataset(train_X, label=train_y)
'''
params = {'task': 'train',
               'boosting': 'gbdt',
               'application': 'multiclass',
               'num_class': 4,
                'metric': 'multi_logloss',
                # 'min_data_in_leaf':500,
                 'num_leaves': 31,
                 'learning_rate': 0.05,
                 'feature_fraction': 1.0,
                 'bagging_fraction': 1.0,
                 'bagging_freq': 2,
               'bagging_seed': 5048,
               'feature_fraction_seed': 2048
}
'''
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_error',
    'max_depth' : 10,
    'num_leaves': 300,
    'min_data_in_leaf': 100,
    'learning_rate': 0.1,
    'feature_fraction': 0.8, # 建树特征选择比例
    'bagging_fraction': 0.8, # 建树样本采样比例
    'bagging_freq': 5, # 每k次迭代执行bagging
    'lambda_l1': 0.4,
    'lambda_l2': 0.5,
    'min_gain_to_split': 0.2,
    'verbose': 5, # >1 显示信息
    'is_unbalance': True
}

print('start training')
# bst = lgb.train(params, lgb_train, num_boost_round=1000)
bst = lgb.train(params,  lgb_train,  num_boost_round=10000)
print('finished')

# 开始预测
predict_y = bst.predict(test_X)
predict_y = predict_y.tolist()
print(predict_y)
predict_Y = []
for i in range(1000):
    # print(predict_y[i])
    num = predict_y[i].index(max(predict_y[i]))
    # print(num)
    predict_Y.append(num)
print(predict_Y)

score = f1_score(test_y, predict_Y, average = 'macro')
print("score: %f" % score)
