import os
import sys
import pickle
import codecs
root_path = os.path.abspath('.')

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

# global variable
PAD_ID = 0
_GO="_GO"
_END="_END"
_PAD="_PAD"


def __csv2label__():
    train_label  = pd.read_csv('../input/train_first.csv')
    train_hankcs = pd.read_csv('../input/train_word.csv')
    train = pd.merge(train_label, train_hankcs, on = 'Id', how = 'left')
    words = train['words'].values
    label = train['Score'].values
    with open('../input/tourist.zh.label.txt', 'a+', encoding='utf-8') as f:
        for i, word in enumerate(words):
            word = word[1:-1]
            word = word.replace(';', ' ')
            word += '\t' + '__label__' + str(label[i]) + '\n'
            f.write(word)


def create_voabulary(word2vec_vocabulary_path, name_scope=''):
    cache_path ='cache_vocabulary_label_pik/'+ name_scope + "_word_voabulary.pik"
    cache_path = os.path.join(root_path, cache_path)
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word = pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}

        vocabulary_word2index['PAD_ID']=0
        vocabulary_index2word[0]='PAD_ID'

        special_index = 0
        with open(word2vec_vocabulary_path, 'r', encoding='utf-8') as f:
            for _, line in enumerate(f.readlines()):
                vocab = line.split(' ')[0]
                vocabulary_word2index[vocab] = _ + 1 + special_index
                vocabulary_index2word[_ + 1 + special_index] = vocab

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index, vocabulary_index2word), data_f)
    return vocabulary_word2index, vocabulary_index2word

# create vocabulary of lables. label is sorted. 1 is high frequency, 2 is low frequency.
def create_voabulary_label(vocabulary_label, regression_flag, name_scope='', use_seq2seq=False):
    print("create_voabulary_label_sorted.started.traning_data_path:",vocabulary_label)
    cache_path ='cache_vocabulary_label_pik/'+ name_scope + "_label_voabulary.pik"
    cache_path = os.path.join(root_path, cache_path)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label=pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label
    else:
        train = codecs.open(vocabulary_label, 'r', 'utf-8')
        lines = train.readlines()

        vocabulary_word2index_label = {}
        vocabulary_index2word_label = {}
        vocabulary_label_count_dict = defaultdict(int)

        for i, line in enumerate(lines):
            if '__label__' in line:  #'__label__-2051131023989903826
                label = line[line.index('__label__') + len('__label__'):].strip().replace("\n","")
                vocabulary_label_count_dict[label] += 1

        list_label = sort_by_value(vocabulary_label_count_dict)
        print("length of list_label:",len(list_label))
        count=0

        ##########################################################################################
        if use_seq2seq: #if used for seq2seq model,insert two special label(token):_GO AND _END
            i_list = [0, 1, 2]
            label_special_list = [_GO, _END, _PAD]
            for _, label in enumerate(label_special_list):
                vocabulary_word2index_label[label] = i_list[_]
                vocabulary_index2word_label[i_list[_]] = label
        #########################################################################################
        if regression_flag:
            print('vocabulary_index2word_label is used as a classification regression')
            for label in list_label:
                label = int(label)
                vocabulary_word2index_label[label] = label
                vocabulary_index2word_label[label] = label
        else:
            for i, label in enumerate(list_label):
                if i < 10:
                    count_value = vocabulary_label_count_dict[label]
                    print("label:", label, "count_value:", count_value)
                    count = count + count_value
                index = i + 3 if use_seq2seq else i
                if use_seq2seq: print('use_seq2seq, please check the code...')
                vocabulary_word2index_label[label] = index
                vocabulary_index2word_label[index] = label
        print("count top10:", count)

        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index_label,vocabulary_index2word_label), data_f)

    print("create_voabulary_label_sorted.ended.len of vocabulary_label:", len(vocabulary_index2word_label))
    return vocabulary_word2index_label, vocabulary_index2word_label

def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0,len(backitems))]

def load_data_multilabel_new(vocabulary_word2index,
                             vocabulary_word2index_label,
                             using_kfold=False,
                             valid_portion=0.3,
                             max_training_data=1000000,
                             training_data_path='../input/tourist.zh.train.txt',
                             multi_label_flag=False,
                             use_seq2seq=False,
                             seq2seq_label_length=6):  # n_words=100000,
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1. load a tourist data from file
    # example: "userid 好 大 的 一个 游乐 公园 已经 去 了 2 次 但 感觉 还 没有 玩 够 似的 会 有 第 三 第 四 次 的	__label__5"
    print("load_data.started...")
    print("load_data_multilabel_new.training_data_path:", training_data_path)
    data_f = codecs.open(training_data_path, 'r', 'utf8')
    lines = data_f.readlines()

    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    Y_decoder_input=[] #ADD 2017-06-15
    ID = []
    for i, line in enumerate(lines):
        data, y = line.split('__label__')
        y = y.strip().replace('\n', '')

        data = data.strip()

        if i < 1:
            print(i, "x0:", data)

        data = data.split(" ")
        id = data[0:1]
        x = data[1:]
        # if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        x = [vocabulary_word2index.get(e, 0) for e in x]
        if i < 2:
            print(i, "x1:", x)

        # 1) prepare label for seq2seq format(ADD _GO,_END,_PAD for seq2seq)
        if use_seq2seq:
            ys = y.replace('\n', '').split(" ")  # ys is a list
            _PAD_INDEX = vocabulary_word2index_label[_PAD]
            ys_mulithot_list = [_PAD_INDEX] * seq2seq_label_length
            ys_decoder_input = [_PAD_INDEX] * seq2seq_label_length
            # below is label.
            for j, y in enumerate(ys):
                if j < -1:
                    ys_mulithot_list[j]=vocabulary_word2index_label[y]
            if len(ys)>seq2seq_label_length-1:
                ys_mulithot_list[seq2seq_label_length-1]=vocabulary_word2index_label[_END]# ADD END TOKEN
            else:
                ys_mulithot_list[len(ys)] = vocabulary_word2index_label[_END]

            # below is input for decoder.
            ys_decoder_input[0]=vocabulary_word2index_label[_GO]
            for j, y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_decoder_input[j+1]=vocabulary_word2index_label[y]
            if i < 10:
                print(i,"ys:==========>0", ys)
                print(i,"ys_mulithot_list:==============>1", ys_mulithot_list)
                print(i,"ys_decoder_input:==============>2", ys_decoder_input)
        else:
            # 2) prepare multi-label format for classification
            if multi_label_flag:
                ys = y.replace('\n', '').split(" ")  # ys is a list
                ys_index = []
                for y in ys:
                    y_index = vocabulary_word2index_label[y]
                    ys_index.append(y_index)
                ys_mulithot_list = transform_multilabel_as_multihot(ys_index)
            else:
                # 3) prepare single label format for classification
                ys_mulithot_list = vocabulary_word2index_label[y]
        if i <= 3:
            print("ys_index:")
            print(i, "y:", y, " ;ys_mulithot_list:", ys_mulithot_list)

        X.append(x)
        Y.append(ys_mulithot_list)
        ID.append(id)

        if use_seq2seq:
            Y_decoder_input.append(ys_decoder_input) #decoder input

    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:", number_examples)

    if using_kfold:
        X = np.array(X)
        Y = np.array(Y)

        kf = list(StratifiedKFold(n_splits=5, random_state=2018, shuffle=False).split(X, Y))
        for train_index, test_index in kf:
            ID_ = []
            for index in test_index:
                ID_.append(ID[index][0])

            yield X[train_index], Y[train_index], X[test_index], Y[test_index], ID_
    else:
        X, X_, y, y_ = train_test_split(X, Y, test_size=valid_portion, random_state=0)
        X, y = data_augmentation(X, y, method=None)
        number_examples = len(X)
        print("number_examples:", number_examples)

        train = (X, y)
        valid = (X_, y_)

        if use_seq2seq:
            train = train + (Y_decoder_input[0:int((1 - valid_portion) * number_examples)],)
            test = valid + (Y_decoder_input[int((1 - valid_portion) * number_examples) + 1:],)

        # 5.return
        print("load_data.ended...")
        yield train, valid, valid

def data_augmentation(X, y, method = None):
    if method is not None:
        return method(X, y)
    else:
        return X, y

def shuffle(X, y):
    X = X.copy()
    y = y.copy()

    X_ = []
    y_ = []
    for x, label in zip(X, y):
        X_.append(np.random.permutation(x[::]))
        y_.append(label)

    X = np.concatenate((X, X_), axis = 0)
    y = np.concatenate((y, y_), axis = 0)
    return X, y

def transform_multilabel_as_multihot(label_list):
    """
    :param label_list: e.g.[0,1,4]
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result = np.zeros(len(label_list))
    result[label_list] = 1
    return result


def load_final_test_data(filename):
    with open(filename, 'r', encoding='utf-8') as outf:
        question_lists_result = []
        for line in outf.readlines():
            line = line.replace('\n', '')
            id, val = line.split(' ')[0], line.split(' ')[1:]
            question_lists_result.append((id, val))
        print("length of total question lists:", len(question_lists_result))
        return question_lists_result

def process_one_sentence_to_get_ui_bi_tri_gram(sentence, n_gram = 3):
    """
    :param sentence: string. example:'w17314 w5521 w7729 w767 w10147 w111'
    :param n_gram:
    :return:string. example:'w17314 w17314w5521 w17314w5521w7729 w5521 w5521w7729 w5521w7729w767 w7729 w7729w767 w7729w767w10147 w767 w767w10147 w767w10147w111 w10147 w10147w111 w111'
    """
    result=[]
    word_list=sentence.split(" ") #[sentence[i] for i in range(len(sentence))]
    unigram='';bigram='';trigram='';fourgram=''
    length_sentence=len(word_list)
    for i,word in enumerate(word_list):
        unigram=word                           #ui-gram
        word_i=unigram
        if n_gram>=2 and i+2<=length_sentence: #bi-gram
            bigram="".join(word_list[i:i+2])
            word_i=word_i+' '+bigram
        if n_gram>=3 and i+3<=length_sentence: #tri-gram
            trigram="".join(word_list[i:i+3])
            word_i = word_i + ' ' + trigram
        if n_gram>=4 and i+4<=length_sentence: #four-gram
            fourgram="".join(word_list[i:i+4])
            word_i = word_i + ' ' + fourgram
        if n_gram>=5 and i+5<=length_sentence: #five-gram
            fivegram="".join(word_list[i:i+5])
            word_i = word_i + ' ' + fivegram
        result.append(word_i)
    result=" ".join(result)
    return result

def load_data_predict(vocabulary_word2index, vocabulary_word2index_label, questionid_question_lists, uni_to_tri_gram=False):
    final_list = []
    for i, tuple in enumerate(questionid_question_lists):
        id, question_string_list = tuple
        # question_string_list = question_string_list[1:-1]
        # question_string_list = question_string_list.replace(";"," ")
        if uni_to_tri_gram:
            x_ = process_one_sentence_to_get_ui_bi_tri_gram(question_string_list)
            x = x_.split(" ")
        else:
            # x = question_string_list.split(" ")
            x = question_string_list
        x = [vocabulary_word2index.get(e, 0) for e in x]
        if i <= 2:
            print("question_id:", id)
            print("question_string_list:", question_string_list)
            print("x_indexed:", x)

        final_list.append((id, x))

    number_examples = len(final_list)
    print("number_examples:",number_examples) #
    return final_list

def split_data(filename = '../input/tourist.zh.label.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        train_outf = open(filename.replace('label', 'train'), 'w', encoding='utf-8')
        test_outf = open(filename.replace('label', 'test'), 'w', encoding='utf-8')
        for line in f.readlines():
            label = line[line.index('__label__') + len('__label__'):].strip().replace("\n", "")
            if label != '-1':
                train_outf.write(line)
            else:
                test_outf.write(line[:line.index('__label__')].strip() + '\n')
    pass


# vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_vocabulary_path='../utils/dump/vocabulary', name_scope="TextCNN")
# vocab_size = len(vocabulary_word2index)
# vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(vocabulary_label='../input/tourist.zh.label.txt', name_scope="TextCNN")
# final_list = load_data_predict(vocabulary_word2index, vocabulary_word2index_label, load_final_test_data('../input/predict_word.csv'))
# print(final_list)


if __name__ == '__main__':
    # split_data()

    stacking_0 = pd.read_csv('../models/TextCNN/__models__/TextCNN_TFIDF_tra_0.csv')
    stacking_1 = pd.read_csv('../models/TextCNN/__models__/TextCNN_TFIDF_tra_1.csv')
    stacking_2 = pd.read_csv('../models/TextCNN/__models__/TextCNN_TFIDF_tra_2.csv')
    stacking_3 = pd.read_csv('../models/TextCNN/__models__/TextCNN_TFIDF_tra_3.csv')
    stacking_4 = pd.read_csv('../models/TextCNN/__models__/TextCNN_TFIDF_tra_4.csv')
    stacking = pd.concat([stacking_0, stacking_1, stacking_2, stacking_3, stacking_4])

    stacking.to_csv('TextCNN_train.csv', index = False)

    test_0 = pd.read_csv('../models/TextCNN/__models__/tourist_result_cnn_multilabel_v6_e14_kf_0.csv')
    test_1 = pd.read_csv('../models/TextCNN/__models__/tourist_result_cnn_multilabel_v6_e14_kf_1.csv')
    test_2 = pd.read_csv('../models/TextCNN/__models__/tourist_result_cnn_multilabel_v6_e14_kf_2.csv')
    test_3 = pd.read_csv('../models/TextCNN/__models__/tourist_result_cnn_multilabel_v6_e14_kf_3.csv')
    test_4 = pd.read_csv('../models/TextCNN/__models__/tourist_result_cnn_multilabel_v6_e14_kf_4.csv')

    test_0 = pd.merge(test_0, test_1, on = 'Id', how = 'left')
    test_0 = pd.merge(test_0, test_2, on = 'Id', how = 'left')
    test_0 = pd.merge(test_0, test_3, on = 'Id', how='left')
    test_0 = pd.merge(test_0, test_4, on = 'Id', how='left')

    Id = test_0[['Id']]
    Id['textCNN_label'] = test_0.drop(['Id'], axis=1).mean(axis=1)

    Id.to_csv('TextCNN_test.csv', index = False)