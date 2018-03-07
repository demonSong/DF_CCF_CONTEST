"""
@author: DemonSong
@time: 2018-02-12 09:42
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import time

import numpy as np
import pandas as pd
from models.get_datasets import load_datasets, threshold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm

from utils.lgbm_feature_selector import filter_zero_weights, greedy_filter

class Bagging(object):
    def __init__(self, clf, params, kfold, folds, num_round, early_stopping_rounds):
        self.clf = clf
        self.params = params
        self.kfold = kfold
        self.folds = folds
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds

    def __inner_train(self, clf, train_x, train_y, test_x, clf_name = 'lgb', class_num=1):

        train = np.zeros((train_x.shape[0], class_num))
        test = np.zeros((test_x.shape[0], class_num))
        test_pre = np.empty((self.folds, test_x.shape[0], class_num))
        cv_scores = []

        for i, (train_index, test_index) in enumerate(self.kfold):
            tr_x = train_x[train_index]
            tr_y = train_y[train_index]
            te_x = train_x[test_index]
            te_y = train_y[test_index]

            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y, reference=train_matrix)

            if test_matrix:
                model = clf.train(self.params,
                                  train_matrix,
                                  num_boost_round=self.num_round,
                                  valid_sets=test_matrix,
                                  early_stopping_rounds=self.early_stopping_rounds)
                pre = model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0], 1))
                train[test_index] = pre
                test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0], 1))
                cv_scores.append(rmse(te_y, pre))
            print ("%s now score is:" % clf_name, cv_scores)

        test[:]=test_pre.mean(axis=0)
        print ("%s_score_list:" % clf_name, cv_scores)
        print ("%s_score_mean:" % clf_name, np.mean(cv_scores))
        return train.reshape(-1, class_num), test.reshape(-1, class_num), np.mean(cv_scores)

    def train(self, X_train, y_train, X_test):
        y_train_pred, y_test_pred, cv_scores = self.__inner_train(self.clf, X_train, y_train, X_test, clf_name='lgb')
        return y_train_pred, y_test_pred, cv_scores

def rmse(y_train, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_train, y_pred))

# 3, 4, 5 随机划分4份
def bacth(train, batch_size = 4):
    train_score3 = train[train['Score'] == 3]
    train_score4 = train[train['Score'] == 4]
    train_score5 = train[train['Score'] == 5]
    train_other = train[(train['Score'] == 1) | (train['Score'] == 2)]

    # shuffle
    train_score3 = shuffle(train_score3, random_state=0)
    train_score4 = shuffle(train_score4, random_state=0)
    train_score5 = shuffle(train_score5, random_state=0)

    for i in range(batch_size):
        train_3 = generate(train_score3, i, batch_size)
        train_4 = generate(train_score4, i, batch_size)
        train_5 = generate(train_score5, i, batch_size)
        train_all = pd.concat([train_3, train_4, train_5, train_other])
        yield train_all

def generate(train, i, batch_size):
    size = train.shape[0] // batch_size
    return train[i * size : (i + 1) * size]

def filter(train, test):
    # features = filter_zero_weights('split_importance_02181349.csv')
    features = greedy_filter('sub_feature_02182137.csv')
    features = features[:180]
    print('features: ', features)
    return train[features], test[features]

def sub_train(train, test, hasFilter = False):
    X_train = train.drop(['Id', 'Score'], axis=1)
    X_test = test.drop(['Id'], axis=1)

    if hasFilter:
        print('filter training...')
        X_train, X_test = filter(X_train, X_test)

    y_train = train['Score']
    df_columns = X_train.columns
    print('特征数 %d' % len(df_columns), X_train.shape[0])

    # transform data structure
    X_train = np.array(X_train.values)
    X_test = np.array(X_test.values)
    y_train = y_train.values

    # define
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'min_child_weight': 20,
        'num_leaves': 2 ** 5,
        'lambda_l2': 2,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'learning_rate': 0.01,
        'seed': 2017,
        'verbose': 100,
        'silent': True,
    }

    folds = 5
    # kf = KFold(X_train.shape[0], n_folds=folds, shuffle=True, random_state=seed)
    # kf = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(X_train))
    kf = list(StratifiedKFold(n_splits=5, random_state=2018, shuffle=True).split(X_train, y_train))

    bagging = Bagging(clf=lightgbm,
                      params=params,
                      folds=folds,
                      kfold=kf,
                      num_round=5000,
                      early_stopping_rounds=100)

    y_train_pred, y_test_pred, cv_scores = bagging.train(X_train, y_train, X_test)

    mean_train_rmse = rmse(y_train_pred, train['Score'].values)

    mean_train_rmse = 1 / (1 + mean_train_rmse)
    mean_test_rmse = 1 / (1 + cv_scores)
    print('mean_train_rmse: ', mean_train_rmse, 'mean_test_rmse: ', mean_test_rmse)
    return y_test_pred, mean_test_rmse

def main():
    print('load train test datasets')
    train, test = load_datasets(dropDuplicate=False)
    batch_size = 1
    submit_df = pd.DataFrame({'userid': test['Id']})

    submit_pred = np.zeros((test.shape[0], 1))
    submit_pred_n = np.zeros((test.shape[0], batch_size))

    test_rmses = []
    for _, train_all in enumerate(bacth(train, batch_size)):
        print('第 %d 批 dataset 开始训练' % _)
        y_test_pred, mean_test_rmse = sub_train(train_all, test, hasFilter=False)
        submit_pred_n[:, _] = y_test_pred.reshape(-1)
        test_rmses.append(mean_test_rmse)
        print('第 %d 批 dataset 训练结束' % _)

    print('train finished...')

    test_rmse = np.mean(test_rmses)
    print('mean test rmse: ', test_rmse)
    submit_pred[:] = submit_pred_n.mean(1).reshape(-1, 1)

    # test
    submit_df['Score'] = submit_pred
    submit_df['Score'] = submit_df['Score']
    submission_path_raw = '../result/{}_{}_{}.csv'.format('lgb_raw', test_rmse,
                                                      time.strftime('%m%d%H%M',
                                                                    time.localtime(time.time())))

    submission_path_threshold = '../result/{}_{}_{}.csv'.format('lgb_threshold', test_rmse,
                                                          time.strftime('%m%d%H%M',
                                                                        time.localtime(time.time())))
    submit_df.to_csv(submission_path_raw, index=False, header=False)
    submit_df = threshold(submit_df)
    submit_df.to_csv(submission_path_threshold, index=False, header=False)

    print('done.')


if __name__ == '__main__':
    print('============lgb training============')
    main()