import pandas as pd
from gensim import models
import os
import jieba

from collections import defaultdict
from sklearn import metrics

def word2vector(path='../models/word2vec_txt.txt', model_name = '300features_5minwords_10context.model'):
    # 加载模型，根据保持时的格式不同，有两种加载方式

    # model = models.Word2Vec.load(os.path.join('..', 'models', model_name))
    # model_txt = models.KeyedVectors.load_word2vec_format(os.path.join('..', 'models', 'word2vec_txt.txt'), binary=False)
    #
    # # 可以同时取出一个句子中单词的词向量
    # wv = model_txt.wv[['鼠标', '天下', '好的']]

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    word_list = []
    vector_list = []
    info = lines[0]
    # word_ids = dict()
    word_num = len(lines)-1
    word_ids = defaultdict(lambda :(word_num-1))
    id_words = dict()
    for i, line in enumerate(lines[1:]):
        array_list = line.strip('\n').split(' ')
        word = array_list[0]
        word_ids[word] = i
        id_words[i] = word
        vector = array_list[1:]
        vector = list(map(float, vector))
        word_list.append(word)
        vector_list.append(vector)

    return word_ids, id_words, vector_list
    # datas = pd.read_csv(path, names=['word', 'embedding'], sep=' ')
    # #构建字典
    # a = 1

def getSentence(path):
    datas = pd.read_csv(path, names=['cat', 'label', 'text'], sep=',')
    sentences = datas['text'][0:].tolist()
    sentences_seg = []
    with open('../datas/chineseStopWords.txt', encoding='utf-8') as f:
        lines = f.readlines()
    chinese_stopwords = []
    for line in lines:
        stopword = line.strip('\n')
        chinese_stopwords.append(stopword)
    for s in sentences:
        # 对于每一个原始句子，进行如下处理
        if (type(s) != str):
            continue
        # 使用jieba分词
        sent = list(jieba.cut(sentence=s, cut_all=False))
        # 去除停用词
        sent = [w for w in sent if (w not in chinese_stopwords and w.strip() != '')]
        sentences_seg.append(sent)
    return sentences_seg


def model_eval(forest, train_data, target):
    print("\n准确率、召回率和F1值为：\n")
    pred = forest.predict(train_data)
    print(metrics.classification_report(target, pred))
