"""
@author: DemonSong
@time: 2018-02-09 17:33

"""

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from models.get_datasets import load_datasets
from sklearn import metrics


def rmse(y_train, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_train, y_pred))

def main():
    print('load train test datasets')
    train, test = load_datasets()

    submit_df = pd.DataFrame({'userid' : test['Id']})

    X_train = train.drop(['Id','Score'], axis = 1)
    X_test  = test.drop(['Id'], axis = 1)

    y_train = train['Score'] - 1
    df_columns = X_train.columns

    xgb_params = {
        'eta': 0.01,
        'min_child_weight': 20,
        'colsample_bytree': 0.5,
        'max_depth': 10,
        'subsample': 0.5,
        'lambda': 2.0,
        'scale_pos_weight': 1,
        'eval_metric': 'mlogloss',
        'objective': 'multi:softmax',
        'silent': 1,
        'booster': 'gbtree',
        'num_class': 5
    }

    dtrain_all = xgb.DMatrix(X_train.values, y_train.values, feature_names=df_columns)
    dtest = xgb.DMatrix(X_test.values, feature_names=df_columns)

    # 5 折交叉验证
    nfold = 5
    cv_result = xgb.cv(dict(xgb_params),
                       dtrain_all,
                       nfold=nfold,
                       stratified=True,
                       num_boost_round=10000,
                       early_stopping_rounds=100,
                       verbose_eval=100,
                       show_stdv=False)

    best_num_boost_rounds = len(cv_result)
    mean_train_mlogloss = cv_result.loc[best_num_boost_rounds - 11: best_num_boost_rounds - 1, 'train-mlogloss-mean'].mean()
    mean_test_mlogloss = cv_result.loc[best_num_boost_rounds - 11: best_num_boost_rounds - 1, 'test-mlogloss-mean'].mean()
    print('best_num_boost_rounds = {}'.format(best_num_boost_rounds))


    # num_boost_round = int(best_num_boost_rounds * 1.1)
    # print('num_boost_round = ', num_boost_round)

    print('mean_rmse_auc = {:.7f} , mean_rmse_auc = {:.7f}\n'.format(mean_train_mlogloss,
                                                                     mean_test_mlogloss))

    print('---> training on total dataset')
    model = xgb.train(dict(xgb_params),
                      dtrain_all,
                      num_boost_round=best_num_boost_rounds)

    print('---> predict test')
    y_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    submit_df['Score'] = y_pred
    submit_df['Score'] = submit_df['Score'] + 1
    print(y_pred)
    submission_path = '../result/{}_{}.csv'.format('xgb', mean_test_mlogloss)

    submit_df.to_csv(submission_path, index=False, header = False)
    print('done.')


if __name__ == '__main__':
    print('============xgboost training============')
    main()