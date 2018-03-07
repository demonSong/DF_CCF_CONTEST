import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
import warnings
warnings.filterwarnings("ignore")

import re
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics

def build_dataset(words, vocabulary_size = 5000):
    from collections import Counter
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(vocabulary_size - 1))
    w_dictionary = {}
    for word, _ in count:
        w_dictionary[word] = len(w_dictionary)
    da = list()
    unk_count = 0
    for word in words:
        if word in w_dictionary:
            index = w_dictionary[word]
        else:
            index = 0
            unk_count += 1
        da.append(index)
    count[0][1] = unk_count
    reverse_dictionary = {zip(w_dictionary.values(), w_dictionary.keys())}
    return da, count, w_dictionary, reverse_dictionary

def rmsel(true_label,pred):
    rmse = np.sqrt(metrics.mean_squared_error(true_label, pred))
    return 1 / (1 + rmse)


train = pd.read_csv('../input/train_first.csv')
predict = pd.read_csv('../input/predict_first.csv')
predict['Score'] = -1

data = pd.concat([train, predict])
data.head()


stop_word = []
stop_words_path = '../input/stop_word.txt'
with open(stop_words_path,encoding='utf-8') as f:
    for line in f.readlines():
        stop_word.append(line.strip())
stop_word.append(' ')

def clean_str(stri):
    stri = re.sub(r'[a-zA-Z0-9]+','',stri)
    cut_str = jieba.cut(stri.strip())
    list_str = [word for word in cut_str if word not in stop_word]
    return list_str

data['words'] = data['Discuss'].apply(lambda x : clean_str(x))
data.head()

d2v_train = data['words'].copy()
d2v_train.head()

all_words = []
for i in d2v_train:
    all_words.extend(i)
print(all_words[0:100])

da, count, w_dictionary, reverse_dictionary = build_dataset(all_words, vocabulary_size = len(all_words))
print(count[0:100])


def get_sent(x, dictionary):
    encode = []
    for i in x:
        if i in dictionary:
            encode.append(dictionary[i])
        else:
            encode.append(0)
    return encode


data['sent'] = data['words'].apply(lambda x: get_sent(x, w_dictionary))
data.head()

train_df = data[data['Score'] != -1]
predict_df = data[data['Score'] == -1]
del predict_df['Score']

train_df.head()

maxlen = 10
print("Pad sequences (samples x time)")

train_df['sent'] = list(sequence.pad_sequences(train_df['sent'], maxlen=maxlen))
predict_df['sent'] = list(sequence.pad_sequences(predict_df['sent'], maxlen=maxlen))

nfolds = 5


def training(train_df, train_label, test_df):
    X = np.array(list(train_df['sent']))
    y = np.array(np_utils.to_categorical(train_label))
    T = np.array(list(test_df['sent']))
    folds = list(StratifiedKFold(n_splits=nfolds, random_state=2018, shuffle=True).split(X, train_label.values))

    S_train = np.zeros((X.shape[0], 1))  # 训练样本数 * 模型个数
    S_test = np.zeros((T.shape[0], 1))  # 测试集样本数 * 模型个数
    S_test_n = np.zeros((T.shape[0], len(folds)))  # 测试集样本数 * n_folds

    error = []
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X[train_idx]  # 训练集特征
        y_train = y[train_idx]  # 训练集标签

        X_holdout = X[test_idx]  # 待预测的输入
        y_holdout = y[test_idx]

        print('Build model...')
        model = Sequential()
        model.add(Embedding(len(w_dictionary) + 1, 256))
        model.add(LSTM(256))  # try using a GRU instead, for fun
        model.add(Dropout(0.5))
        model.add(Dense(6))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=32, nb_epoch=2, validation_data=(X_holdout, y_holdout))

        y_true = [np.argmax(i) for i in list(y_holdout)]
        predictions = list(model.predict(X_holdout, batch_size=32))
        y_pred = [np.sum(i * [0, 1, 2, 3, 4, 5]) for i in predictions]
        print('rmse: {}'.format(rmsel(y_true, y_pred)))
        error.append(rmsel(y_true, y_pred))

        submission = list(model.predict(T, batch_size=32))
        sub_pred = [np.sum(i * [0, 1, 2, 3, 4, 5]) for i in submission]

        S_train[test_idx] = np.array(y_pred).reshape(-1, 1)
        S_test_n[:, j] = np.array(sub_pred)

    S_test[:] = S_test_n.mean(1).reshape(-1, 1)
    return S_train, S_test, round(np.mean(error), 5)


S_train, S_test, error = training(train_df, train_df['Score'], predict_df)


train_out = train_df[['Id']]
train_out['lstm_len_10'] = S_train
train_out.to_csv('../models/__models__/train_lstm_len_10.csv', index = False)

test_out = predict_df[['Id']]
test_out['lstm_len_10'] = S_test

test_out.to_csv('../models/__models__/test_lstm_len_10.csv', index = False)

print('error: {}'.format(error))
