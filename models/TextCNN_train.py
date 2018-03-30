# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from models.TextCNN.TextCNN_model import TextCNN
from utils.load_dataset import create_voabulary, create_voabulary_label, load_data_multilabel_new
from gensim.models import Word2Vec as word2vec
from conf.configure import Configure
from sklearn.metrics import mean_squared_error

from tflearn.data_utils import to_categorical, pad_sequences
import os

#configuration
FLAGS=tf.app.flags.FLAGS

# self_config
tf.app.flags.DEFINE_boolean("regression_flag", False, "whether to use regression to predict or not")
tf.app.flags.DEFINE_integer("num_classes",5,"number of label") # 改成回归模型

tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.65, "Rate of decay for learning rate.") #0.65一次衰减多少
#tf.app.flags.DEFINE_integer("num_sampled",50,"number of noise sampling") #100
tf.app.flags.DEFINE_string("ckpt_dir","text_cnn_title_desc_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",20,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",200,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",11,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
#tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")

#训练文件存放的地方
tf.app.flags.DEFINE_string("training_data_path","train-zhihu4-only-title-all.txt","path of traning data.") #O.K.train-zhihu4-only-title-all.txt-->training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
tf.app.flags.DEFINE_integer("num_filters", 512, "number of filters") #256--->512
#word2vec存放路径
tf.app.flags.DEFINE_string("word2vec_model_path","../../utils/dump/word2vec_gensim.model","word2vec's vocabulary and vectors") #zhihu-word2vec.bin-100-->zhihu-word2vec-multilabel-minicount15.bin-100
tf.app.flags.DEFINE_boolean("multi_label_flag", False,"use multi label or single label.")

# 干嘛用的
filter_sizes=[1,2,3,4,5,6,7]

# 1.load data(X:list of lint,y:int)
# 2.create session
# 3.feed data
# 4.training
# 5.validation
# 6.prediction

def main(_):
    # load data
    trainX, trainY, testX, testY = None, None, None, None
    vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_vocabulary_path='../../utils/dump/vocabulary', name_scope="TextCNN")
    vocab_size = len(vocabulary_word2index)
    print("cnn_model.vocab_size:", vocab_size)

    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(regression_flag=FLAGS.regression_flag, vocabulary_label='../../input/tourist.zh.train.txt', name_scope="TextCNN")
    # train, test, _ = load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label, using_kfold=True,
    #                                           training_data_path='../../input/tourist.zh.train.txt',
    #                                           multi_label_flag=FLAGS.multi_label_flag)
    kf_id = -1
    for train_X, train_y, valid_X, valid_y, ID in load_data_multilabel_new(vocabulary_word2index,
                                                                       vocabulary_word2index_label,
                                                                       using_kfold= True,
                                                                       training_data_path='../../input/tourist.zh.train.txt',
                                                                       multi_label_flag=FLAGS.multi_label_flag):
        kf_id += 1
        if kf_id != 4: continue
        trainX, trainY = train_X, train_y
        testX, testY = valid_X, valid_y
        print('hello', ID)

        # 2. Data preprocessing.Sequence padding
        print("start padding & transform to one hot...")
        trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
        testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
        print("trainX[0]:", trainX[0])

        # Converting labels to binary vectors
        print("end padding & transform to one hot...")

        #2.create session.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            #Instantiate Model
            textCNN = TextCNN(filter_sizes,FLAGS.num_filters,FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                            FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training, regression_flag=FLAGS.regression_flag,  multi_label_flag=FLAGS.multi_label_flag)

            train_write = tf.summary.FileWriter('log/train_{}'.format(kf_id), sess.graph)
            test_write = tf.summary.FileWriter('log/test_{}'.format(kf_id))
            merged = tf.summary.merge_all()

            #Initialize Save
            saver = tf.train.Saver()
            if os.path.exists(FLAGS.ckpt_dir+str(kf_id) + "/" + "checkpoint_{}".format(kf_id)):
                print("Restoring Variables from Checkpoint")
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            else:
                print('Initializing Variables')
                sess.run(tf.global_variables_initializer())
                if FLAGS.use_embedding:  #load pre-trained word embedding
                    assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN, word2vec_model_path=FLAGS.word2vec_model_path)

            curr_epoch = sess.run(textCNN.epoch_step)
            # 3.feed data & training
            number_of_training_data = len(trainX)
            batch_size = FLAGS.batch_size
            index = 0
            for epoch in range(curr_epoch, FLAGS.num_epochs):
                loss, acc, counter = 0.0, 0.0, 0
                # 批处理
                for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                    if epoch==0 and counter == 0:
                        print("trainX[start:end]:", trainX[start:end])#;print("trainY[start:end]:",trainY[start:end])
                    feed_dict = {textCNN.input_x: trainX[start:end],textCNN.dropout_keep_prob: 0.5}
                    if not FLAGS.multi_label_flag:
                        feed_dict[textCNN.input_y] = trainY[start:end]
                    else:
                        feed_dict[textCNN.input_y_multilabel] = trainY[start:end]
                    summary, curr_loss, curr_acc, _ = sess.run([merged, textCNN.loss_val, textCNN.accuracy, textCNN.train_op], feed_dict) #curr_acc--->TextCNN.accuracy
                    loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc
                    if counter % 50 == 0:
                        print("Epoch %d\tBatch %d\tTrain Loss:%.5f\tTrain Accuracy:%.5f" %(epoch,counter,loss/float(counter),acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
                    train_write.add_summary(summary, index)
                    index += 1
                #epoch increment
                print("going to increment epoch counter....")
                sess.run(textCNN.epoch_increment)

                # 4.validation
                print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
                if epoch % FLAGS.validate_every==0:
                    eval_loss, eval_acc=do_eval(sess,merged,test_write, index, textCNN,testX,testY,batch_size,vocabulary_index2word_label)
                    print("Epoch %d Validation Loss:%.5f\tValidation Accuracy: %.5f" % (epoch,eval_loss,eval_acc))
                    #save model to checkpoint
                    save_path=FLAGS.ckpt_dir+ str(kf_id) + "/" + "model.ckpt"
                    saver.save(sess,save_path,global_step=epoch)


            # 5. 最后在测试集上做测试，并报告测试准确率 Test
            # test_loss, test_acc = do_eval(sess, merged,test_write,epoch, textCNN, testX, testY, batch_size, vocabulary_index2word_label)
            # print("Validation Loss:%.5f\tValidation Accuracy: %.5f" % (test_loss, test_acc))

            # 6. 自定义衡量指标
            self_acc = _eval(sess, textCNN, testX, testY, vocabulary_index2word_label, ID=ID, kf_id=kf_id, regression_flag=FLAGS.regression_flag, namse_scope = 'demon')

        train_write.close()
    pass

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textCNN, word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    word2vec_model = word2vec.load(word2vec_model_path)
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.wv.vocab, word2vec_model.wv.vectors):
        word2vec_dict[word] = vector

    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# report my loss
def _rmse(y, y_):
    error = np.sqrt(mean_squared_error(y, y_))
    return 1 / (1 + error)

def _eval(sess, textCNN, evalX, evalY, vocabulary_index2word_label, regression_flag, ID, kf_id, namse_scope = ''):
    batch_size = 1
    number_examples = len(evalX)
    print(number_examples, len(ID))
    y = []
    y_ = []
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples + 1, batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1}
        logits = sess.run(textCNN.logits, feed_dict)
        if not regression_flag:
            predicted_labels, value_labels = get_label_using_logits_with_value(logits[0], vocabulary_index2word_label)
            value_labels_exp = np.exp(value_labels)
            p_labels = value_labels_exp / np.sum(value_labels_exp)
            pred_ = get_single_value(predicted_labels, p_labels, threshold = False)
        else:
            pred_ = logits[0][0]
        if len(y_) % 5000 == 0: print(pred_)
        y_.append(pred_)
        y.append(vocabulary_index2word_label[evalY[start:end][0]])

    print(len(y_))
    acc = _rmse(np.array(y).astype('float64'), np.array(y_).astype('float64'))

    # result
    with open('__models__/TextCNN_TFIDF_tra_{}.csv'.format(kf_id), 'w') as outf:
        for _ in range(len(ID)):
            outf.write('{},{}'.format(ID[_], y_[_]) + '\n')

    print(namse_scope, ' rmse accuracy: {}'.format(acc))
    return acc

# get label using logits
def get_label_using_logits_with_value(logits, vocabulary_index2word_label, top_number=5):
    index_list = np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list = index_list[::-1]
    value_list = []
    label_list = []
    for index in index_list:
        label = vocabulary_index2word_label[index]
        label_list.append(label)
        value_list.append(logits[index])
    return label_list, value_list

def get_single_value(predict_labels, p_labels, threshold = True):
    label = 0.0
    for i, j in zip(predict_labels, p_labels):
        label += int(i) * j
    if threshold:
        if label >= 4 and label < 4.732: label = 4
        elif label >= 4.732: label = 5
        elif label <= 1: label = 1
    return label

# 在验证集上做验证，报告损失、精确度
def do_eval(sess, merged, test_write ,index, textCNN, evalX, evalY, batch_size, vocabulary_index2word_label):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1}
        if not FLAGS.multi_label_flag:
            feed_dict[textCNN.input_y] = evalY[start:end]
        else:
            feed_dict[textCNN.input_y_multilabel] = evalY[start:end]
        summary, curr_eval_loss, logits, curr_eval_acc = sess.run([merged, textCNN.loss_val, textCNN.logits, textCNN.accuracy], feed_dict)#curr_eval_acc--->textCNN.accuracy
        test_write.add_summary(summary, index + eval_counter)
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)

#从logits中取出前五 get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=1):
    #print("get_label_using_logits.logits:",logits) #1-d array: array([-5.69036102, -8.54903221, -5.63954401, ..., -5.83969498,-5.84496021, -6.13911009], dtype=float32))
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    #label_list=[]
    #for index in index_list:
    #    label=vocabulary_index2word_label[index]
    #    label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return index_list

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

if __name__ == "__main__":
    tf.app.run()