"""
@author: DemonSong
@time: 2018-02-09 17:33

"""

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from conf.configure import Configure

def load_datasets(filename = None, dropDuplicate = False, fillNan = False):
    print('load baseline features')

    # 加载特征
    if filename is not None:
        data = pd.read_csv(filename)
    else:
        data = pd.read_csv(Configure.root_feature_path)

    train = data[data['Score'] != -1]
    test  = data[data['Score'] == -1]
    test = test.drop(['Score'], axis = 1)

    if fillNan:
        train.fillna(train.median(), inplace=True)
        test.fillna(test.median(), inplace=True)

    if dropDuplicate:
        user = pd.read_csv(Configure.root_data_path + 'train_first.csv')
        user.drop_duplicates(subset='Discuss', keep='first',inplace=True)
        train = pd.merge(user[['Id']], train, on = 'Id', how = 'left')

    return train, test
# ratio = [938, 2189, 3792, 9874, 13207]
def load_stacking_datasets(sample = False, dropDuplicate = False, ratio = []):
    print('load stacking features')

    # 加载特征
    data = pd.read_csv(Configure.root_stacking_path)

    train = data[data['Score'] != -1]
    if sample and len(ratio) == 5:
        print('sample...')
        all_len = len(train)
        # 5
        test_ratio = ratio[4] / sum(ratio)
        sub_train = train[train.Score == 5].reset_index(drop=True)
        score_5_df = sub_train.sample(n=int(test_ratio * all_len), replace=True)

        # 4
        test_ratio = ratio[3] / sum(ratio)
        sub_train = train[train.Score == 4].reset_index(drop=True)
        score_4_df = sub_train.sample(n=int(test_ratio * all_len), replace = True)

        # 3
        test_ratio = ratio[2] / sum(ratio)
        sub_train = train[train.Score == 3].reset_index(drop=True)
        score_3_df = sub_train.sample(n=int(test_ratio * all_len), replace = True)

        # 2
        test_ratio = ratio[1] / sum(ratio)
        sub_train = train[train.Score == 2].reset_index(drop=True)
        score_2_df = sub_train.sample(n=int(test_ratio * all_len), replace = True)

        # 1
        test_ratio = ratio[0] / sum(ratio)
        sub_train = train[train.Score == 1].reset_index(drop=True)
        score_1_df = sub_train.sample(n=int(test_ratio * all_len), replace = True)

        train = pd.concat([score_5_df, score_4_df, score_3_df, score_2_df, score_1_df])

    if dropDuplicate:
        user = pd.read_csv(Configure.root_data_path + 'train_first.csv')
        user.drop_duplicates(subset='Discuss', keep='first',inplace=True)
        train = pd.merge(user[['Id']], train, on = 'Id', how = 'left')

    test  = data[data['Score'] == -1]
    test = test.drop(['Score'], axis = 1)
    return train, test

def random_choice(df, all_len, test_ratio):
    choice = np.random.choice(len(all_len), int(test_ratio * all_len), replace=False)
    df = df[choice]
    return df

def pre_process(train, test):
    X_train = train.drop(['Id', 'Score'], axis=1)
    X_test = test.drop(['Id'], axis=1)
    y_train = train['Score']
    df_columns = X_train.columns
    print('特征数 %d' % len(df_columns), X_train.shape[0])
    return X_train, y_train, X_test, df_columns



def split_data(data, label):
    data_classify_5 = data.copy()
    data_classify_5['Score'].ix[(data_classify_5['Score'] != label) & (data_classify_5['Score'] != -1)] = 0
    data_classify_5['Score'].ix[data_classify_5['Score'] == label] = 1
    data_classify_5.to_csv(Configure.root_multi2binary_path + 'data_{}.csv'.format(label), index = False)
    print('done...')


def get_multi2binary_datasets(label, fillNan = False):
    """
    :param label: 1, 2, 3, 4, 5
    :return:
    """
    data = pd.read_csv(Configure.root_multi2binary_path + 'data_{}.csv'.format(label))
    train = data[data['Score'] != -1]
    test = data[data['Score'] == -1]

    test = test.drop(['Score'], axis=1)
    if fillNan:
        train.fillna(train.median(), inplace=True)
        test.fillna(test.median(), inplace=True)

    return train, test

def threshold(result, feature='Score'):
    boolean = (result[feature] >= 4.0) & (result[feature] < 4.732)
    result[feature].ix[boolean] = 4
    result[feature].ix[(result[feature] >= 4.732)] = 5.0
    result[feature].ix[result[feature] < 1] = 1.0
    return result

if __name__ == '__main__':
    train, test = load_stacking_datasets(dropDuplicate=True)
    print(train.info(), train.shape)