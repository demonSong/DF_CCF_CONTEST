# _*_coding:utf-8 _*_
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jieba
import re
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn import metrics
import random

import fasttext


data_path = '../input/train_first.csv'
df = pd.read_csv(data_path,header = 0, encoding='utf8')

test_data_path = '../input/predict_first.csv'
test_df = pd.read_csv(test_data_path,header = 0, encoding='utf8')

stop_word = []
stop_words_path = '../input/stop_word.txt'
with open(stop_words_path,encoding='utf8') as f:
    for line in f.readlines():
        stop_word.append(line.strip())
stop_word.append(' ')

def clean_str(stri):
    stri = re.sub(r'[a-zA-Z0-9]+','',stri)
    cut_str = jieba.cut(stri.strip())
    list_str = [word for word in cut_str if word not in stop_word]
    stri = ' '.join(list_str)
    return stri

df['Discuss'] = df['Discuss'].map(lambda x : clean_str(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x : clean_str(x))

def fillnull(x):
    if x == '':
        return '空白'
    else:
        return x

df['Discuss'] = df['Discuss'].map(lambda x: fillnull(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x: fillnull(x))

def fasttext_data(data,label):
    out = pd.DataFrame({
        'data': data,
        'label': label
    })
    def to_label(x):
        return '__label__{}'.format(x)
    out['label'] = out['label'].apply(lambda x : to_label(x))
    out.to_csv('../input/train.txt', index = False, header = False, sep = '\t', encoding='utf-8')
    return '../input/train.txt'

def get_predict(pred):
    score = np.array([1,2,3,4,5])
    pred2 = []
    for p in pred:
        pr = np.sum(p * score)
        pred2.append(pr)
    return np.array(pred2)

def rmsel(true_label,pred):
    rmse = np.sqrt(metrics.mean_squared_error(true_label, pred))
    return 1 / (1 + rmse)


def fast_cv(df):
    X = df['Discuss'].values
    y = df['Score'].values
    fast_pred = []
    folds = list(KFold(n_splits=5, shuffle=True, random_state=2018).split(X, y))
    for train_index, test_index in folds:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_file = fasttext_data(X_train, y_train)
        classifier = fasttext.supervised(train_file, '../input/model', lr=0.01, dim=128, label_prefix="__label__", encoding = 'utf-8-sig')
        result = classifier.predict_proba(df.loc[test_index, 'Discuss'].tolist(), k=5)
        print(result[0:100])
        pred = [[int(sco) * proba for sco, proba in result_i] for result_i in result]
        pred = [sum(pred_i) for pred_i in pred]
        print(pred[0:100])
        print(rmsel(y_test, pred))

        test_result = classifier.predict_proba(test_df['Discuss'].tolist(), k=5)
        fast_predi = [[int(sco) * proba for sco, proba in result_i] for result_i in test_result]
        fast_predi = [sum(pred_i) for pred_i in fast_predi]
        fast_pred.append(fast_predi)

    fast_pred = np.array(fast_pred)
    fast_pred = np.mean(fast_pred, axis=0)
    return fast_pred

test_pred1 = fast_cv(df)
print(test_pred1[0:100])
