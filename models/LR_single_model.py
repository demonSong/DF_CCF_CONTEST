from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import time

import numpy as np

from models.get_datasets import load_stacking_datasets
from conf.configure import Configure

def rmse_(true_label,pred):
    rmse = np.sqrt(mean_squared_error(true_label, pred))
    return 1 / (1 + rmse)


def threshold(result):
    boolean = (result['Score'] >= 4.0) & (result['Score'] < 4.732)
    result['Score'].ix[boolean] = 4
    result['Score'].ix[(result['Score'] >= 4.732)] = 5.0
    result['Score'].ix[result['Score'] < 1] = 1.0
    return result

def lr_training(X, y, T):
    clf = LogisticRegression()
    clf.fit(X, y)

    y_ = clf.predict(X)
    rmse = rmse_(y_, y)

    print(clf.coef_)
    submit = clf.predict(T)
    return submit, rmse


def main():
    train, test = load_stacking_datasets()
    X = train.drop(['Id', 'Score'], axis=1)
    y = train['Score']

    print(X.shape)

    T = test.drop(['Id'], axis = 1)
    pred_, rmse = lr_training(X, y, T)
    print('error : {}'.format(rmse))

    sumbit = test[['Id']]
    sumbit['Score'] = pred_

    submit = threshold(sumbit)

    submission_path = '../result/{}_{}_{}.csv'.format('lr_stacking', rmse,
                                                          time.strftime('%m%d%H%M',
                                                                        time.localtime(time.time())))
    submit.to_csv(submission_path, index = False, header = False)


if __name__ == '__main__':
    main()