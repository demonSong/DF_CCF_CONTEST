from sklearn import linear_model
from models.get_datasets import load_datasets

import time
import numpy as np

from conf.configure import Configure
from models.stacking import Ensemble
from models.lgbm_stacking_reg import getDataSet

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def ridge_train(X, y, test, features, clf, kBest = False,k = 2, minMaxScaler = None, ratio = 1):

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

    CVRmse = 1 / (1 + CVRmse)

    return CVRmse

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def grid_search(X, y):

    clf = Ridge(max_iter = 500, tol = 0.01, fit_intercept=True)
    param_dist = {"alpha": np.logspace(-3,2,20),
                  'normalize':[True, False]}

    grid_search = GridSearchCV(clf, param_grid=param_dist, cv=5)
    grid_search.fit(X, y)

    report(grid_search.cv_results_, n_top=5)


if __name__ == '__main__':
    train, test = load_datasets(fillNan=True)
    X_train, y_train, X_test, df_coulumns = getDataSet(train, test)

    X = X_train[df_coulumns].values
    y = y_train

    X = np.array(X)
    y = np.array(y)

    select = SelectKBest(f_regression, k=476)
    X = select.fit_transform(X, y)
    grid_search(X, y)
