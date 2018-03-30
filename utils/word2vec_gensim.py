import gensim
import logging
import multiprocessing
import sys
from time import time
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class Sentences(object):
    def __init__(self, filename, encoding='utf-8'):
        self.filename = filename
        self.encoding = encoding
 
    def __iter__(self):
        with open(self.filename, 'r', encoding=self.encoding) as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                yield line

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Please use python word2vec_gensim.py data_path")
        exit()
    data_path = sys.argv[1]

    begin = time()
    sentences = Sentences(data_path)
    model = gensim.models.Word2Vec(sentences,
                                   size=200,
                                   window=10,
                                   min_count=10, # 过滤频率 < min_count 的词
                                   workers=multiprocessing.cpu_count())

    model.save("dump/word2vec_gensim.model")
    model.wv.save_word2vec_format("dump/word2vec_org",
                                  "dump/vocabulary",
                                  binary=False)

    end = time()
    print ("Total procesing time: %d seconds" % (end - begin))


