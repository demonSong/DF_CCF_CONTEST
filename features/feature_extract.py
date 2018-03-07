# 导包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import jieba

from collections import Counter
from snownlp import SnowNLP

import collections
import os

from conf.configure import Configure

def senti_snownlp(dicuss):
    if dicuss == '': return 0.5
    return SnowNLP(dicuss).sentiments

def senti_sequence_arra(word, window=3):
    sent = word[1:-1]
    words = sent.split(';')
    buffer = collections.deque(maxlen=window)
    arra = []
    for i in range(window):
        if i < len(words):
            buffer.append(words[i])
    arra.append(senti_snownlp(''.join(list(buffer))))
    for i in range(window, len(words)):
        buffer.append(words[i])
        arra.append(senti_snownlp(''.join(list(buffer))))
    return arra


def calFeatureWithSentiWindow(data_word, feature_word='words', window=3):
    words = data_word[feature_word].values
    feature_max = []
    feature_min = []
    feature_std = []
    feature_avg = []
    feature_median = []
    cnt = 0
    for word in words:
        arra = senti_sequence_arra(word, window=window)
        if len(arra) == 0:
            feature_max.append('')
            feature_min.append('')
            feature_std.append('')
            feature_avg.append('')
            feature_median.append('')
        else:
            feature_max.append(np.max(arra))
            feature_min.append(np.min(arra))
            feature_std.append(np.std(arra))
            feature_avg.append(np.mean(arra))
            feature_median.append(np.median(arra))
        if cnt % 500 == 0: print(cnt)
        cnt += 1

    print(feature_max[0:100])
    data_word['senti_max_window_{}_{}'.format(window, feature_word)] = np.array(feature_max).reshape(-1, 1)
    data_word['senti_min_window_{}_{}'.format(window, feature_word)] = np.array(feature_min).reshape(-1, 1)
    data_word['senti_std_window_{}_{}'.format(window, feature_word)] = np.array(feature_std).reshape(-1, 1)
    data_word['senti_avg_window_{}_{}'.format(window, feature_word)] = np.array(feature_avg).reshape(-1, 1)
    data_word['senti_median_window_{}_{}'.format(window, feature_word)] = np.array(feature_median).reshape(-1, 1)

    return data_word

def runSubData():
    data_word = pd.read_csv(Configure.root_sub_data_path + 'train_word.csv')
    for w in [1, 3, 5, 7]:
        data_word = calFeatureWithSentiWindow(data_word, feature_word='words', window=w)
    del data_word['words']
    data_word.to_csv(Configure.root_sub_data_path + 'data_senti_window_hancks.csv', index=False)

    # jieba 分词
    data_jieba = pd.read_csv(Configure.root_sub_data_path + 'data_jieba.csv')
    for w in [1, 3, 5, 7]:
        data_jieba = calFeatureWithSentiWindow(data_jieba, feature_word='words_jieba', window=w)
    del data_jieba['words_jieba']
    data_jieba.to_csv(Configure.root_sub_data_path + 'data_senti_window_jieba.csv', index=False)

def runAllData():
    # hancks 分词
    train_word = pd.read_csv('../input/train_word.csv')
    test_word = pd.read_csv('../input/predict_word.csv')
    data_word = pd.concat([train_word, test_word])
    data_word.head()

    for w in [1, 3, 5, 7]:
        data_word = calFeatureWithSentiWindow(data_word, feature_word='words', window=w)


    del data_word['words']
    data_word.to_csv('../features/data_senti_window_hancks.csv', index = False)

    # jieba 分词
    data_jieba = pd.read_csv('../input/data_jieba.csv')

    for w in [1, 3, 5, 7]:
        data_jieba = calFeatureWithSentiWindow(data_jieba, feature_word='words_jieba', window=w)

    del data_jieba['words_jieba']
    data_jieba.to_csv('../features/data_senti_window_jieba.csv', index = False)


if __name__ == '__main__':
    runSubData()