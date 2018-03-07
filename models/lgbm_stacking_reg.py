from models.stacking import Ensemble
from models.get_datasets import load_datasets, pre_process, get_multi2binary_datasets,threshold
from conf.configure import Configure

import pandas as pd
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge,Lasso

from lightgbm.sklearn import LGBMRegressor, LGBMClassifier

def getDataSet(train, test):
    X_train, y_train, X_test, df_coulumns = pre_process(train, test)
    return train, y_train, test, df_coulumns

def training_regression(X, y, test, features, clfs, kBest = False,
                        k = 2, minMaxScaler = None, ratio = 1, model_name = []):
    global_time = time.strftime("%m%d%H%M", time.localtime())
    for turn, clf in enumerate(clfs):
        startTime = int(time.time())
        print('第', turn, ' 轮训练开始...')

        choosen_col = []
        choosen = np.random.choice(len(features), int(ratio * len(features)), replace=False)

        for index_ in choosen:
            choosen_col.append(features[index_])

        train_feature = X[choosen_col].values
        test_feature = test[choosen_col].values

        if minMaxScaler is not None:
            print('MinMaxScaler exist.')
            train_feature = minMaxScaler.fit_transform(train_feature)
            test_feature = minMaxScaler.transform(test_feature)

        stacking = Ensemble(5, clf, seed=2018, kBest=kBest,k=k)
        S_train, S_test, CVRmse, CVStd = stacking.fit_predict(train_feature, y, test_feature)  # 输出的是列向量

        train_out = X[['Id']]
        train_out[model_name[turn]] = S_train

        train_out.to_csv(Configure.root_model_stacking_path + 'train_' + model_name[turn] + '.csv', index=False)
        print('train persist done...')

        test_out = test[['Id']]
        test_out[model_name[turn]] = S_test
        test_out.to_csv(Configure.root_model_stacking_path + 'test_' + model_name[turn] + '.csv', index=False)
        print('test persist done...')

        CVRmse = 1 / (1 + CVRmse)
        with open(Configure.root_model_stacking_path + 'cvMap_{}.csv'.format(global_time), 'a+') as outf:
            outf.write('{},{},{}'.format(turn, CVRmse, CVStd) + '\n')
        print('error: {}'.format(CVRmse))
        endTime = int(time.time())
        print('第', turn, ' 轮训练结束...所花时间:' + str((endTime - startTime) / 60) + ' 分钟')

def training_classification(X, y, test, features, clfs, ratio = 1, model_name = []):
    global_time = time.strftime("%m%d%H%M", time.localtime())
    for _, clf in enumerate(clfs):
        print('第 {} 个 分类器开始预测'.format(_))
        for turn, model in enumerate(model_name):
            startTime = time.strftime("%m%d%H%M", time.localtime())
            print('第', turn,' 轮训练开始...开始时间: ', startTime)

            choosen_col = []
            choosen = np.random.choice(len(features), int(ratio * len(features)), replace=False)

            for index_ in choosen:
                choosen_col.append(features[index_])

            train_feature = X[choosen_col].values
            test_feature = test[choosen_col].values

            # 缩放
            scaler = MinMaxScaler((-1, 1))
            train_feature = scaler.fit_transform(train_feature)
            test_feature = scaler.transform(test_feature)

            stacking = Ensemble(5, clf, seed = 2018, metric='auc')
            S_train, S_test, CVRmse, CVStd = stacking.fit_predict(train_feature, y, test_feature, predict_proba = True)  # 输出的是列向量

            train_out = X[['Id']]
            train_out[model] = S_train

            train_out.to_csv(Configure.root_model_stacking_path + 'train_clf_{}_label_{}.csv'.format(_, model), index = False)
            print('train persist done...')

            test_out = test[['Id']]
            test_out[model] = S_test
            test_out.to_csv(Configure.root_model_stacking_path + 'test_clf_{}_label_{}.csv'.format(_, model), index = False)
            print('test persist done...')

            with open(Configure.root_model_stacking_path + 'cvMap_{}.csv'.format(global_time), 'a+') as outf:
                outf.write('{},{},{}'.format(turn, CVRmse, CVStd) + '\n')

            endTime = time.strftime("%m%d%H%M", time.localtime())
            print('第', turn, ' 轮训练结束...所花时间:' + str(int(endTime) - int(startTime)) + ' 分钟')

def training_lgb_classification(X, y, test, features, ratio = 1, model_name = []):
    global_time = time.strftime("%m%d%H%M", time.localtime())

    for turn, model in enumerate(model_name):
        startTime = time.strftime("%m%d%H%M", time.localtime())
        print('第', turn,' 轮训练开始...开始时间: ', startTime)

        choosen_col = []
        choosen = np.random.choice(len(features), int(ratio * len(features)), replace=False)

        for index_ in choosen:
            choosen_col.append(features[index_])

        train_feature = X[choosen_col].values
        test_feature = test[choosen_col].values

        depth = 6 + np.random.randint(-2, 3)
        lgbm = LGBMClassifier(
            n_estimators = 5000,
            task = 'train',
            min_child_weight= 20,
            boosting_type = 'gbdt',
            objective = 'binary',
            metric ='auc',
            num_leaves= 2 ** (depth + 1),
            learning_rate= 0.01,
            feature_fraction= np.random.randint(3, 9) / 10, # same as colsample_bytree
            bagging_fraction= np.random.randint(3, 9) / 10, # same as subsample
            reg_lambda = np.random.randint(2, 11),
            seed = np.random.randint(0, 2018)
        )

        stacking = Ensemble(5, lgbm, seed = 2018, metric='auc')
        S_train, S_test, CVRmse, CVStd = stacking.fit_predict(train_feature, y, test_feature, predict_proba = True)  # 输出的是列向量

        train_out = X[['Id']]
        train_out[model] = S_train

        train_out.to_csv(Configure.root_model_stacking_path + 'train_lgbmc_' + model + '.csv', index = False)
        print('train persist done...')

        test_out = test[['Id']]
        test_out[model] = S_test
        test_out.to_csv(Configure.root_model_stacking_path + 'test_lgbmc_' + model + '.csv', index = False)
        print('test persist done...')

        with open(Configure.root_model_stacking_path + 'cvMap_{}.csv'.format(global_time), 'a+') as outf:
            outf.write('{},{},{}'.format(turn, CVRmse, CVStd) + '\n')

        endTime = time.strftime("%m%d%H%M", time.localtime())
        print('第', turn, ' 轮训练结束...所花时间:' + str(int(endTime) - int(startTime)) + ' 分钟')

def training_lgbm_regressor(X, y, test, features, ratio = 1, model_name = []):
    global_time = time.strftime("%m%d%H%M", time.localtime())
    for turn, model in enumerate(model_name):
        startTime = time.strftime("%m%d%H%M", time.localtime())
        print('第', turn,' 轮训练开始...开始时间: ', startTime)

        choosen_col = []
        choosen = np.random.choice(len(features), int(ratio * len(features)), replace=False)

        for index_ in choosen:
            choosen_col.append(features[index_])

        train_feature = X[choosen_col].values
        test_feature = test[choosen_col].values

        depth = 6 + np.random.randint(-2, 3)
        lgbm = LGBMRegressor(
            n_estimators = 5000,
            task = 'train',
            min_child_weight= 20,
            boosting_type = 'gbdt',
            objective = 'regression',
            metric ='rmse',
            num_leaves= 2 ** (depth + 1),
            learning_rate= 0.01,
            feature_fraction= np.random.randint(3, 9) / 10, # same as colsample_bytree
            bagging_fraction= np.random.randint(3, 9) / 10, # same as subsample
            reg_lambda = np.random.randint(2, 11),
            seed = np.random.randint(0, 2018)
        )

        stacking = Ensemble(5, lgbm, seed = 2018)
        S_train, S_test, CVRmse, CVStd = stacking.fit_predict(train_feature, y, test_feature)  # 输出的是列向量


        train_out = X[['Id']]
        train_out[model] = S_train

        train_out.to_csv(Configure.root_model_stacking_path + 'train_lgbmr_' + model + '.csv', index = False)
        print('train persist done...')

        test_out = test[['Id']]
        test_out[model] = S_test
        test_out.to_csv(Configure.root_model_stacking_path + 'test_lgbmr_' + model + '.csv', index = False)
        print('test persist done...')

        result = threshold(test_out, feature=model)
        result.to_csv('../result/lgbmr_{}_{}.csv'.format(model, time.strftime("%m%d%H%M", time.localtime())),
                                                         index = False, header = False)

        CVRmse = 1 / (1 + CVRmse)
        with open(Configure.root_model_stacking_path + 'cvMap_{}.csv'.format(global_time), 'a+') as outf:
            outf.write('{},{},{}'.format(turn, CVRmse, CVStd) + '\n')

        endTime = time.strftime("%m%d%H%M", time.localtime())
        print('第', turn, ' 轮训练结束...所花时间:' + str(int(endTime) - int(startTime)) + ' 分钟')





def classification():

    clfs = [RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='entropy')]
            # GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=10, n_estimators=5000)]

    for label in range(3, 6):
        train, test = get_multi2binary_datasets(label, fillNan=True)
        X_train, y_train, X_test, df_coulumns = getDataSet(train, test)

        models = ['lc_label_{}'.format(label)]
        training_classification(X_train, y_train, X_test, df_coulumns, clfs, ratio = 1, model_name = models)

def regression():

    train, test = load_datasets(fillNan=True)
    X_train, y_train, X_test, df_coulumns = getDataSet(train, test)

    model_name = ['ridge_1', 'ridge_2', 'lasso_1', 'lasso_2']
    clfs = [Ridge(fit_intercept=True, alpha=8.858667904100823, max_iter=500, normalize=False, tol=0.01),
            Ridge(fit_intercept=True, alpha=8.858667904100823, max_iter=500, normalize=True,  tol=0.01),
            Lasso(fit_intercept=True, alpha=8.858667904100823, max_iter=500, normalize=True,  tol=0.01),
            Lasso(fit_intercept=True, alpha=8.858667904100823, max_iter=500, normalize=False, tol=0.01)]

    training_regression(X_train, y_train, X_test, df_coulumns, clfs, kBest=True, k=476, ratio=1, model_name=model_name)

    model_name = ['ridge_3', 'ridge_4', 'lasso_3', 'lasso_4']
    training_regression(X_train, y_train, X_test, df_coulumns, clfs, minMaxScaler= MinMaxScaler((-1, 1)), kBest=True, k=476, ratio=1, model_name=model_name)


def lgbm_regressor():
    train, test = load_datasets()
    X_train, y_train, X_test, df_coulumns = getDataSet(train, test)
    models = ['lr_' + str(i) for i in range(201, 300)]
    training_lgbm_regressor(X_train, y_train, X_test, df_coulumns, ratio = 0.8, model_name = models)

if __name__ == '__main__':
    lgbm_regressor()