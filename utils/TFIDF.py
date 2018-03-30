import os
import sys
from collections import defaultdict

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
import numpy as np

from conf.configure import Configure
from utils.filter import get_wordSet, read_map

def get_discuss_map(data_word, feature = 'words'):
    users = data_word['Id'].values
    discuss = data_word[feature].values
    discussMap = defaultdict(set)
    for _, user in enumerate(users):
        words = discuss[_]
        words = words[1:-1]
        words = words.split(';')
        for word in words:
            discussMap[user].add(word)
    return discussMap

def idf(data_word, feature = 'words'):
    dictionary = get_wordSet(data_word, feature)
    discussMap = get_discuss_map(data_word, feature=feature)

    with open(Configure.root_data_path + 'idf_{}.map'.format(feature), 'a', encoding='utf-8') as f:
        for _, word in enumerate(dictionary):
            cnt = 0
            for key, val in discussMap.items():
                if word in val:
                    cnt += 1
            if _ % 500 == 0:
                print(_, word, np.log(len(dictionary) / cnt))
            f.write('{} {}'.format(word, np.log(len(dictionary) / cnt)) + '\n')

# 一个词在这个文档上的重要程度
def get_top_tf_idf_map(topK = 100):
    # load data
    data_train = pd.read_csv(Configure.root_data_path + 'train_word.csv')
    data_predict = pd.read_csv(Configure.root_data_path + 'predict_word.csv')
    data_word = pd.concat([data_train, data_predict])

    if not os.path.exists(Configure.root_data_path + 'idf.map'):
        idf(data_word, feature='words')
    else:
        idf_map = read_map('idf.map')

    discuss = data_word['words'].values

    tf_idf_map = defaultdict(float)
    for _, dis in enumerate(discuss):
        tf_map = defaultdict(int)
        words = dis[1:-1]
        words = words.split(';')
        for word in words:
            tf_map[word] += 1
        n = len(words)

        for key in tf_map.keys():
            if key in idf_map:
                if key in tf_idf_map:
                    val = tf_idf_map[key]
                    tf_idf_map[key] = max(val, tf_map[key] / n * float(idf_map[key]))
                else:
                    tf_idf_map[key] = tf_map[key] / n * float(idf_map[key])
        if _ % 5000 == 0: print(_, dis)

    sorted_tf_idf = sorted(tf_idf_map.items(), key=lambda v : v[1], reverse=True)
    return sorted_tf_idf[0:topK]

def read_label_list(filename, label=1):
    with open(Configure.root_data_path + filename, encoding='utf-8') as f:
        dictionary = dict()
        for _, line in enumerate(f.readlines()):
            line = line.strip().split(' ')

        return dictionary


def pad_sequence(maxlen = 20, feature = 'words'):
    data_train = pd.read_csv(Configure.root_data_path + 'train_word.csv')
    data_predict = pd.read_csv(Configure.root_data_path + 'predict_word.csv')
    data_word = pd.concat([data_train, data_predict])

    score = pd.read_csv(Configure.root_data_path + 'data.csv')
    data_word = pd.merge(data_word, score[['Id', 'Score']], on = 'Id', how = 'left')

    if not os.path.exists(Configure.root_data_path + 'idf_{}.map'.format(feature)):
        idf(data_word, feature=feature)
    else:
        idf_map = read_map('idf_{}.map'.format(feature))

    Id = data_word['Id'].values
    discuss = data_word[feature].values
    score = data_word['Score'].values

    topK = maxlen
    key_words = []
    tf_idf_df = []

    with open('../input/tourist.zh.label.txt', 'w', encoding='utf-8') as outf:
        for _, dis in enumerate(discuss):
            tf_map = defaultdict(int)
            words = dis[1:-1]
            words = words.split(';')
            for word in words:
                tf_map[word] += 1
            n = len(words)

            if n == 0:
                print('Error no words, userId is {}', Id[_])
                continue

            tf_idf_map = defaultdict(float)
            for key in tf_map.keys():
                if key in idf_map:
                    tf_idf_map[key] = tf_map[key] / n * float(idf_map[key])

            # get tf-idf map
            tf_idf_max = sorted(tf_idf_map.items(), key=lambda val: val[1], reverse=True)
            top_tf_idf_set = set()
            value = []

            for i in range(min(len(tf_idf_max), maxlen)):
                top_tf_idf_set.add(tf_idf_max[i][0])

            if n >= maxlen: # 需要截断
                for word in words:
                    if word in top_tf_idf_set:
                        value.append(word)
            else:
                for word in words:
                    if word in top_tf_idf_set:
                        value.append(word)
                choice = np.random.choice(len(words), maxlen - len(words), replace=True)
                for c in choice:
                    value.append(words[c])

            if _ % 5000 == 0:
                print(_, value)

            sentence = Id[_] + ' ' + ' '.join(value) + '\t' + '__label__' + str(score[_]) + '\n'
            outf.write(sentence)
    pass


def main():
    # load data
    data_train = pd.read_csv(Configure.root_data_path + 'train_word.csv')
    data_predict = pd.read_csv(Configure.root_data_path + 'predict_word.csv')
    data_word = pd.concat([data_train, data_predict])

    if not os.path.exists(Configure.root_data_path + 'idf.map'):
        idf(data_word, feature='words')
    else:
        idf_map = read_map('idf.map')

    discuss = data_word['words'].values

    topK = 100
    key_words = []
    tf_idf_df = []

    for _, dis in enumerate(discuss):
        tf_map = defaultdict(int)
        words = dis[1:-1]
        words = words.split(';')
        for word in words:
            tf_map[word] += 1
        n = len(words)
        if n == 0:
            key_words.append([''] * topK)
            tf_idf_df.append([0] * topK)
            continue
        tf_idf_map = defaultdict(float)
        for key in tf_map.keys():
            if key in idf_map:
                tf_idf_map[key] = tf_map[key] / n * float(idf_map[key])
        if len(tf_idf_map) == 0:
            key_words.append([''] * topK)
            tf_idf_df.append([0] * topK)
            continue

        # get tf-idf map
        tf_idf_max = sorted(tf_idf_map.items(), key = lambda val : val[1], reverse = True)
        key_word_topK = [''] * topK
        tf_idf_topK = [0] * topK
        for i in range(min(len(tf_idf_max), topK)):
            key_word_topK[i] = tf_idf_max[i][0]
            tf_idf_topK[i] = tf_idf_max[i][1]

        key_words.append(key_word_topK)
        tf_idf_df.append(tf_idf_topK)
        if _ % 5000 == 0: print(_, key_words[max(0, _ - 10):_])


    for i in range(topK):
        data_word['key_word_' + str(i)] = [val[i] for val in key_words]
        data_word['tf_idf_' + str(i)] = [val[i] for val in tf_idf_df]

    del data_word['words']
    data_word.to_csv(Configure.root_data_path + 'data_keyword.csv', index = False, encoding='utf-8')
    print('done...')


if __name__ == '__main__':
    pad_sequence()