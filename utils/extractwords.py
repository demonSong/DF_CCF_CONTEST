# 统计3 4 5 6字出现的次数
import pandas as pd
import pickle
import re
import os

from collections import defaultdict

def clean_str(stri):
    r = '[’，。！ 【】!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    stri = re.sub(r,'',stri)
    return stri

def get_word_len_map(discuss, maxLen):
    count_map = defaultdict(int)
    for sentence in discuss:
        for start in range(0, max(len(sentence) - maxLen, 0)):
            word = sentence[start:start + maxLen]
            count_map[word] += 1
    return count_map

def create_word_len_map_label(data, label):
    discuss = data['Discuss'].values
    filename = '../input/cache/word_len_map_{}'.format(label)
    if os.path.exists(filename):
        print('file {} cache exists ...'.format(filename))
        with open(filename, 'rb') as data_f:
            count_map_3, count_map_4, count_map_5, count_map_6 = pickle.load(data_f)
    else:
        print('file {} no cache... try to generate...'.format(filename))
        count_map_3 = get_word_len_map(discuss, 3)
        count_map_4 = get_word_len_map(discuss, 4)
        count_map_5 = get_word_len_map(discuss, 5)
        count_map_6 = get_word_len_map(discuss, 6)

        with open(filename, 'wb') as data_f:
            pickle.dump((count_map_3, count_map_4, count_map_5, count_map_6), data_f)

    return count_map_3, count_map_4, count_map_5, count_map_6

def create_word_len_map(label=None):
    """
    :param label: 考虑不同label过滤出的词
    :return:
    """
    train = pd.read_csv('../input/train_first.csv')
    test  = pd.read_csv('../input/predict_first.csv')
    test['Score'] = -1

    data = pd.concat([train, test])
    data['Discuss'] = data['Discuss'].apply(lambda x : clean_str(x))
    data_word = data.copy()

    print('extracting  feature start....')

    def top_map(count_map, topK=100):
        sorted_map = sorted(count_map.items(), key=lambda x: x[1], reverse=True)[0:topK]
        return sorted_map[0:topK]

    def contain(sentence, val):
        return 1 if val in sentence else 0

    count = 0
    for label in [0, 1, 2, 3, 4, 5]:
        if label == 0:
            count_map_3, count_map_4, count_map_5, count_map_6 = create_word_len_map_label(data, label)
        else:
            data_label = data[data['Score'] == label]
            count_map_3, count_map_4, count_map_5, count_map_6 = create_word_len_map_label(data_label, label)

        # one hot code
        for count_map in (count_map_3, count_map_4, count_map_5, count_map_6):
            sorted_map = top_map(count_map, topK=100)
            for val, cnt in sorted_map:
                data_word['word_len_feature_{}'.format(count)] = data_word['Discuss'].apply(lambda x : contain(x, val))
                print ('{} done...'.format(count))
                count += 1

    data_word = data_word.drop(['Discuss'], axis = 1)
    data_word.to_csv('../input/data_word_len.csv', index=False)
    print('extracting feature end...')

if __name__ == '__main__':
    create_word_len_map()
    pass
