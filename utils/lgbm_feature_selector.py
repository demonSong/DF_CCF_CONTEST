import os
import sys

from collections import defaultdict

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from models.get_datasets import load_datasets
from conf.configure import Configure

def most_importance(filename):
    feature_df = pd.read_csv(Configure.root_model_info_path + filename)
    feature_names = feature_df.columns
    sort_feature = [val for val in feature_names if val != 'feature_name']
    feature_df = feature_df.sort_values(by=sort_feature, ascending=False)
    return feature_df['feature_name'].values.tolist()

def filter_zero_weights(filename, feature = 'split_importance'):
    feature_df = pd.read_csv(Configure.root_model_info_path + filename)
    feature_df = feature_df[feature_df[feature] > 0]
    return feature_df['feature_name'].values.tolist()

def greedy_filter(filename):
    feature_df = pd.read_csv(Configure.root_model_info_path + filename)
    return feature_df['feature_name'].values.tolist()

def rmse(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))

def main():
    # 加载 数据集
    train, text = load_datasets()

    # 加载 feature importance
    features = most_importance('split_importance_02181349.csv')

    best_feature = []
    maxRmse = 0

    drop_feature = []
    drop_len = 10

    drop_freq = defaultdict(int)
    all_feature = features[::]

    while len(features) != 0:
        choose_feature = features.pop(0)
        candicate_feature = best_feature.copy()
        candicate_feature.append(choose_feature)

        train_feature = train[candicate_feature]
        train_label = train['Score']

        train_feature, valid_feature, train_label, valid_label = train_test_split(train_feature, train_label,
                                                                                  test_size=0.3, random_state=0)
        train_feature = np.array(train_feature.values)
        valid_feature = np.array(valid_feature.values)

        train_label = np.array(train_label).reshape(-1)
        valid_label = np.array(valid_label).reshape(-1)

        lgb_train = lgbm.Dataset(train_feature, label=train_label)
        lgb_eval = lgbm.Dataset(valid_feature, label=valid_label, reference=lgb_train)

        lgbm_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'min_child_weight': 20,
            'num_leaves': 2 ** 5,
            'lambda_l2': 2,
            'subsample': 0.5,
            'colsample_bytree': 1,
            'learning_rate': 0.1,
            'seed': 2017,
            'verbose': 100,
            'silent': True,
        }

        model = lgbm.train(lgbm_params,
                           lgb_train,
                           num_boost_round=5000,
                           valid_sets=lgb_eval,
                           early_stopping_rounds=100)

        valid_pred = model.predict(valid_feature, num_iteration=model.best_iteration)
        valid_rmse = rmse(valid_label, valid_pred)

        valid_rmse = 1 / (1 + valid_rmse)
        if valid_rmse >= maxRmse - 0.000245:
            best_feature.append(choose_feature)
            maxRmse = max(maxRmse, valid_rmse)
            with open('../models/info/sub_feature_02182137.csv', 'a+') as out:
                out.write(str(valid_rmse) + ',' + choose_feature + ',' + str(len(drop_feature)) + '\n')
        else:
            drop_freq[choose_feature] += 1
            drop_feature.append(choose_feature)
            if len(drop_feature) == drop_len:
                for val in drop_feature[::-1]:
                    if drop_freq[val] > 1: continue
                    features.insert(0, val)
                drop_feature = []
                drop_len += drop_len

            with open('../models/info/sub_feature_02182137.log', 'a+') as out:
                out.write('当前 drop 队列长度 {}：'.format(len(drop_feature)) + '\n')
                out.write(' || '.join(drop_feature) + '\n')
                for drop in drop_feature:
                    out.write('{} {}'.format(drop_freq[drop], drop) + '\n')

if __name__ == '__main__':
    main()