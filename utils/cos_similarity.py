from collections import Counter
from collections import defaultdict

import pandas as pd

from conf.configure import Configure
from utils.filter import read_map, read_file_word2set

def get_words(data_df, stop_set = set(), feature = 'words'):
    data_df = data_df[feature].values
    dictionary = []
    for sent in data_df:
        sent = sent[1:-1]
        sent = sent.split(';')
        for word in sent:
            if len(stop_set) == 0:
                if word != '':
                    dictionary.append(word)
            else:
                if word != '' and word not in stop_set:
                    dictionary.append(word)
    return dictionary

def top_tf_idf(data_word, idf_map, topK, stop_set = set(), label = 5):
    data_word_5 = data_word[data_word['Score'] == label]
    words = get_words(data_word_5, stop_set = stop_set, feature='words')
    count = Counter(words).most_common()
    word_len = len(words)
    tf_idf_map = defaultdict(float)
    for key, val in count:
        if key in idf_map:
            tf_idf_map[key] = val / word_len * float(idf_map[key])

    tf_idf_top = sorted(tf_idf_map.items(), key = lambda v : v[1], reverse = True)
    print(tf_idf_top[0:min(topK, len(tf_idf_top))])
    return tf_idf_top[0:min(topK, len(tf_idf_top))]

def cos(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB == 0.0:
        return ''
    else:
        return dot_product / ((normA*normB)**0.5)

def main():
    # load 数据集
    data_train = pd.read_csv(Configure.root_data_path + 'train_word.csv')
    data_test = pd.read_csv(Configure.root_data_path + 'predict_word.csv')

    data_score = pd.read_csv(Configure.root_data_path + 'data.csv')
    data_score = data_score[['Id', 'Score']]
    data_word = pd.concat([data_train, data_test])

    data_word = pd.merge(data_word, data_score, on = 'Id', how = 'left')
    idf_map = read_map(Configure.root_data_path + 'idf.map')

    stop_list = read_file_word2set(Configure.root_data_path + 'stop_word.txt')

    columns = ['Id']
    for topK in [10, 30, 50, 100, 200]:
        for label in range(1, 6):
            label_1 = top_tf_idf(data_word, idf_map, topK, stop_set=stop_list, label=label)
            discuss = data_word['words'].values
            feature_with_label_1 = []
            for _, dis in enumerate(discuss):
                tf_map = defaultdict(int)
                words = dis[1:-1]
                words = words.split(';')
                for word in words:
                    tf_map[word] += 1

                tf_idf_map = defaultdict(float)
                if len(words) != 0:
                    for key, val in tf_map.items():
                        if key in idf_map:
                            tf_idf_map[key] = tf_map[key] / len(words) * float(idf_map[key])

                vector1 = [val[1] for val in label_1]
                vector2 = [0] * topK
                for i in range(topK):
                    key = label_1[i][0]
                    if key in tf_idf_map.keys():
                        vector2[i] = tf_idf_map[key]
                    else:
                        vector2[i] = 0
                feature_with_label_1.append(cos(vector1,vector2))
                if _ % 5000 == 0: print(_)

            featureName = 'cos_simlarity_' + str(label) + '_' + str(topK)
            columns.append(featureName)
            data_word[featureName] = feature_with_label_1

    data_word[columns].to_csv(Configure.root_data_path + 'data_similarity.csv', index = False)

if __name__ == '__main__':
    main()