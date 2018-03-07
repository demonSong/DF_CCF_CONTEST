from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pandas as pd
import re
import jieba
import gc

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()



model_file = '../input/'

# 加载数据
def load_dataset():
    train = pd.read_csv(model_file + 'train_first.csv')
    test  = pd.read_csv(model_file + 'predict_first.csv')

    def clean_str(stri):
        stri = re.sub(r'[a-zA-Z0-9]+', '', stri)
        cut_str = jieba.cut(stri.strip())
        return ' '.join(cut_str)

    train['Discuss'] = train['Discuss'].apply(lambda x : clean_str(x))
    test['Discuss'] = test['Discuss'].apply(lambda x : clean_str(x))

    X = []
    for discuss in train['Discuss'].values:
        X.append(discuss)
    y = train['Score'].values

    T = []
    for discuss in test['Discuss'].values:
        T.append(discuss)

    return train[['Id']], test[['Id']], X, y, T

train_id, test_id, X, y, T = load_dataset()

# Extracting features from the training data using a sparse vectorizer
def stop_words():
    stop_word = []
    stop_words_path = '../input/stop_word.txt'
    with open(stop_words_path, encoding='utf-8') as f:
        for line in f.readlines():
            stop_word.append(line.strip())
    return stop_word

print(stop_words()[0:10])

print('using TfidfVectorizer...')
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words=stop_words())

X_train = vectorizer.fit_transform(X)
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(T)
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

feature_names = vectorizer.get_feature_names()
if feature_names:
    feature_names = np.asarray(feature_names)

print(feature_names[0:10])

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


def benchmark(clf, X, y, X_, y_, T, target_names=['label_1', 'label_2', 'label_3', 'label_4', 'label_5']):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X, y)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = np.sqrt(metrics.mean_squared_error(pred, y_))
    score = 1 / (1 + score)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        # 利用这里的top10 的特征 作为关键词
        topK = 50
        if opts.print_top10 and feature_names is not None:
            print("top {} keywords per class:".format(topK))
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-topK:]
                with open('../input/label1to5_{}_{}.map'.format(str(clf)[0:5], topK), 'a+') as outf:
                    print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
                    outf.write('{} {}'.format(i + 1, ';'.join(feature_names[top10])) + '\n')
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


X, X_, y, y_ = train_test_split(X_train, y, test_size=0.2, random_state=0)

opts.print_top10 = True
opts.print_report = True
opts.print_cm = True

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, X, y, X_, y_, X_test))


# print('=' * 80)
# print("KNN")
# results.append(benchmark(KNeighborsClassifier(n_neighbors=10), X, y, X_, y_, X_test))


# print('=' * 80)
# print("Random Forest")
# results.append(benchmark(RandomForestClassifier(n_estimators=100), X, y, X_, y_, X_test))
#
# gc.collect()
#
# print('=' * 80)
# print("Elastic-Net penalty")
# results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                        penalty="elasticnet"), X, y, X_, y_, X_test))
#
# # Train NearestCentroid without threshold
# print('=' * 80)
# print("NearestCentroid (aka Rocchio classifier)")
# results.append(benchmark(NearestCentroid(), X, y, X_, y_, X_test))
#
# gc.collect()
#
# # Train sparse Naive Bayes classifiers
# print('=' * 80)
# print("Naive Bayes")
# results.append(benchmark(MultinomialNB(alpha=.01), X, y, X_, y_, X_test))
# results.append(benchmark(BernoulliNB(alpha=.01), X, y, X_, y_, X_test))
#
# print('=' * 80)
# print("LinearSVC with L1-based feature selection")
# # The smaller C, the stronger the regularization.
# # The more regularization, the more sparsity.
# results.append(benchmark(Pipeline([
#   ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
#                                                   tol=1e-3))),
#   ('classification', LinearSVC(penalty="l2"))]), X, y, X_, y_, X_test))
#
# gc.collect()
#
# indices = np.arange(len(results))
#
# results = [[x[i] for x in results] for i in range(4)]
#
# clf_names, score, training_time, test_time = results
# training_time = np.array(training_time) / np.max(training_time)
# test_time = np.array(test_time) / np.max(test_time)
#
# plt.figure(figsize=(12, 8))
# plt.title("Score")
# plt.barh(indices, score, .2, label="score", color='navy')
# plt.barh(indices + .3, training_time, .2, label="training time",
#          color='c')
# plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
# plt.yticks(())
# plt.legend(loc='best')
# plt.subplots_adjust(left=.25)
# plt.subplots_adjust(top=.95)
# plt.subplots_adjust(bottom=.05)
#
# for i, c in zip(indices, clf_names):
#     plt.text(-.3, i, c)
#
# plt.show()