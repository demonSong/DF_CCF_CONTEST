import pandas as pd
from conf.configure import Configure

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

def allOneLabel(train, word):
    predict_discuss = train['Discuss'].values
    label = train['Score'].values
    cnt = 0
    label_ = -1
    for i, discuss in enumerate(predict_discuss):
        if word in discuss:
            if label_ == -1: label_ = label[i]
            elif label_ != label[i]: return -1, cnt
            cnt += 1
    return label_, cnt


def read_map(filename):
    with open(Configure.root_data_path + filename, encoding='utf-8') as f:
        dictionary = dict()
        for line in f.readlines():
            line = line.strip().split(' ')
            if len(line) != 2: continue
            dictionary[line[0]] = line[1]
        return dictionary

def read_file_word2set(filename, encoding = 'gbk'):
    with open(filename, 'r', encoding = encoding) as f:
        dictionary = set()
        for line in f.readlines():
            line = line.strip().split(' ')
            for word in line:
                dictionary.add(word)
        return dictionary

def get_wordSet(data_df, feature = 'words'):
    data_df = data_df[feature].values
    dictionary = set()
    for sent in data_df:
        sent = sent[1:-1]
        sent = sent.split(';')
        for word in sent:
            if word != '':
                dictionary.add(word)
    return dictionary

def filter_train():
    data_word_train = pd.read_csv(Configure.root_data_path + 'train_word.csv')
    data_word_test = pd.read_csv(Configure.root_data_path + 'predict_word.csv')
    data_word_hankcs = pd.concat([data_word_train, data_word_test])
    data_word_jieba = pd.read_csv(Configure.root_data_path + 'data_jieba.csv')

    wordSet = get_wordSet(data_word_hankcs, feature='words')
    wordSet = wordSet | get_wordSet(data_word_jieba, feature='words_jieba')

    train = pd.read_csv(Configure.root_data_path + 'train_first.csv')

    with open(Configure.root_data_path + 'filter_map.txt', 'a+', encoding='utf-8') as f:
        for _, word in enumerate(wordSet):
            label_, cnt = allOneLabel(train, word)
            if label_ != -1:
                f.write('{} {} {}'.format(label_, word, cnt) + '\n')
            if _ % 500 == 0: print(_, word, label_)

def read_file_word2map(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        dictionary = dict()
        for line in f.readlines():
            line = line.strip().split(' ')
            dictionary[line[1]] = line[0]
        return dictionary


def filter(data_word, predict, feature = 'word'):
    predict.columns = ['Id', 'Score']
    predict = pd.merge(predict, data_word[['Id', feature]], on = 'Id', how = 'left')

    if not os.path.exists(Configure.root_data_path + 'filter_map.txt'):
        filter_train()
    dictionary = read_file_word2map(Configure.root_data_path + 'filter_map.txt')

    Id = predict['Id'].values
    label = predict['Score'].values
    words = predict[feature].values

    for i, word in enumerate(words):
        word = word[1:-1]
        word = word.split(';')
        if len(word) == 0: continue
        for w in word:
            if w in dictionary:
                label[i] = dictionary[w]
                print(w)
                break

    submit = pd.DataFrame({
        'Id' : Id,
        'Score' : label
    })
    return submit

def filter_fetaure(filename, featurefile, topK):
    data = pd.read_csv(filename)

    feature_df = pd.read_csv(featurefile)
    feature_df = feature_df.sort_values(by='gain_importance', ascending=False)
    feature_importance = feature_df['feature_name'].tolist()[0:topK]

    feature_importance.extend(['Id', 'Score'])
    return data[feature_importance]

if __name__ == '__main__':
    # if not os.path.exists(Configure.root_data_path + 'filter_map.txt'):
    #     filter_train()
    #
    # data_word = pd.read_csv(Configure.root_data_path + 'predict_word.csv')
    # predict = pd.read_csv('../result/lgb_0.6057142255356438_02130837.csv')
    # predict_f = filter(data_word, predict, feature='words')
    # predict_f.to_csv('../result/f_0.6057142255356438_02130837.csv', header=False, index=False)


    data = filter_fetaure('../input/data.csv', featurefile='../models/info/gain_importance_data.csv', topK=560)
    data.to_csv('../input/data_560.csv', index=False)