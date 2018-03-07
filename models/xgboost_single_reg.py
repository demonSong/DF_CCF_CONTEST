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
from models.get_datasets import load_datasets, load_stacking_datasets
from sklearn import metrics


def rmse(y_train, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_train, y_pred))

def threshold(result):
    boolean = (result['Score'] >= 4.0) & (result['Score'] < 4.732)
    result['Score'].ix[boolean] = 4
    result['Score'].ix[(result['Score'] >= 4.732)] = 5.0
    result['Score'].ix[result['Score'] < 1] = 1.0
    return result

def main():
    print('load train test datasets')
    train, test = load_stacking_datasets()

    submit_df = pd.DataFrame({'userid' : test['Id']})

    X_train = train.drop(['Id','Score'], axis = 1)
    X_test  = test.drop(['Id'], axis = 1)

    y_train = train['Score']
    df_columns = X_train.columns
    print('特征数 %d' % len(df_columns))

    xgb_params = {
        'eta': 0.01,
        'min_child_weight': 20,
        'colsample_bytree': 0.5,
        'max_depth': 10,
        'subsample': 0.5,
        'lambda': 2.0,
        'scale_pos_weight': 1,
        'eval_metric': 'rmse',
        'objective': 'reg:linear',
        'seed':2018,
        'tree_method':'gpu_exact',
        'silent': 1,
        'booster': 'gbtree'
    }

    dtrain_all = xgb.DMatrix(X_train.values, y_train.values, feature_names=df_columns)
    dtest = xgb.DMatrix(X_test.values, feature_names=df_columns)

    # 5 折交叉验证
    nfold = 5
    cv_result = xgb.cv(dict(xgb_params),
                       dtrain_all,
                       nfold=nfold,
                       stratified=True,
                       num_boost_round=5000,
                       early_stopping_rounds=100,
                       verbose_eval=100,
                       show_stdv=False)

    best_num_boost_rounds = len(cv_result)
    mean_train_rmse = cv_result.loc[best_num_boost_rounds - 11: best_num_boost_rounds - 1, 'train-rmse-mean'].mean()
    mean_test_rmse = cv_result.loc[best_num_boost_rounds - 11: best_num_boost_rounds - 1, 'test-rmse-mean'].mean()
    print('best_num_boost_rounds = {}'.format(best_num_boost_rounds))


    # num_boost_round = int(best_num_boost_rounds * 1.1)
    # print('num_boost_round = ', num_boost_round)
    # mean_train_rmse = 0.6307557787750947
    # mean_test_rmse = 0.6307557787750947
    # best_num_boost_rounds = 1000

    mean_train_rmse = 1 / (1 + mean_train_rmse)
    mean_test_rmse = 1 / (1 + mean_test_rmse)

    print('mean_train_rmse = {:.7f} , mean_test_rmse = {:.7f}\n'.format(mean_train_rmse,
                                                                     mean_test_rmse))

    print('---> training on total dataset')
    model = xgb.train(dict(xgb_params),
                      dtrain_all,
                      num_boost_round=best_num_boost_rounds)

    print('---> predict test')
    y_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    submit_df['Score'] = y_pred
    submit_df['Score'] = submit_df['Score']
    submission_path_raw = '../result/{}_{}_{}.csv'.format('xgb_raw', mean_test_rmse,
                                                      time.strftime('%m%d%H%M',
                                                                    time.localtime(time.time())))

    submission_path_threshold = '../result/{}_{}_{}.csv'.format('xgb_threshold', mean_test_rmse,
                                                          time.strftime('%m%d%H%M',
                                                                        time.localtime(time.time())))
    submit_df.to_csv(submission_path_raw, index=False, header=False)
    submit_df = threshold(submit_df)
    submit_df.to_csv(submission_path_threshold, index=False, header=False)
    print('done.')


if __name__ == '__main__':
    print('============xgboost training============')
    main()