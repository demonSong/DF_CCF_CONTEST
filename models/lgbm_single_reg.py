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
import lightgbm as lgbm
from models.get_datasets import load_datasets, batch_with_gain_importance
from utils.lgbm_utils import get_weights
from sklearn import metrics


def rmse(y_train, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_train, y_pred))

def threshold(result):
    boolean = (result['Score'] >= 4.0) & (result['Score'] < 4.732)
    result['Score'].ix[boolean] = 4
    result['Score'].ix[(result['Score'] >= 4.732)] = 5.0
    result['Score'].ix[result['Score'] <= 1.0] = 1.0
    return result

def _train(train, test, outf):
    submit_df = pd.DataFrame({'userid': test['Id']})

    X_train = train.drop(['Id', 'Score'], axis=1)
    X_test = test.drop(['Id'], axis=1)

    y_train = train['Score']
    df_columns = X_train.columns
    df_columns = [col for col in df_columns]
    print('特征数 %d' % len(df_columns))

    X_train = np.array(X_train.values)
    X_test = np.array(X_test.values)
    y_train = y_train.values

    lgbm_params = {
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
        'verbose': 1,
        'silent': True,
    }

    dtrain = lgbm.Dataset(X_train, label=y_train, feature_name=df_columns)

    # 5 折交叉验证
    cv_results = lgbm.cv(lgbm_params,
                         dtrain,
                         nfold=5,
                         stratified=True,
                         num_boost_round=5000,
                         early_stopping_rounds=100,
                         verbose_eval=50
                         )

    best_num_boost_rounds = len(cv_results['rmse-mean'])
    mean_test_rmse = np.mean(cv_results['rmse-mean'][best_num_boost_rounds - 6: best_num_boost_rounds - 1])
    print('best_num_boost_rounds = {}'.format(best_num_boost_rounds))

    # mean_test_rmse = 0.6307557787750947
    # best_num_boost_rounds = 5000

    mean_test_rmse = 1 / (1 + mean_test_rmse)

    print('mean_test_rmse = {:.7f}\n'.format(mean_test_rmse))

    print('---> training on total dataset')
    model = lgbm.train(lgbm_params, dtrain, num_boost_round=best_num_boost_rounds)

    print('---> predict test')
    y_pred = model.predict(X_test)
    submit_df['Score'] = y_pred
    submit_df['Score'] = submit_df['Score']
    submission_path = '../result/{}_{}_{}.csv'.format('lgb_single', mean_test_rmse,
                                                      time.strftime('%m%d%H%M',
                                                                    time.localtime(time.time())))

    submit_df = threshold(submit_df)
    submit_df.to_csv(submission_path, index=False, header = False)

    print('---> model info')
    get_weights(model)

    outf.write("{} {} {}".format(time.strftime('%m%d%H%M', time.localtime(time.time())), len(df_columns),
                                mean_test_rmse) + '\n')
    print('done.')

def main():
    print('load train test datasets')
    usebatch = False

    # configuration display
    print('batch is used: {}'.format("Yes" if usebatch else "No"))

    with open('../input/feature.info', 'a+') as outf:
        if usebatch:
            for train, test in batch_with_gain_importance(featurefile='../models/info/gain_importance_data.csv', filename='../input/data.csv'):
                _train(train, test, outf)
        else:
            train, test = load_datasets(filename='../input/data_560.csv')
            _train(train, test, outf)

if __name__ == '__main__':
    print('============lgbm training============')
    main()