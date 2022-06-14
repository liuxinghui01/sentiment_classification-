import os
import pandas as pd

from gensim import models
import jieba
import logging
from util.util import getSentence


"""打印日志信息"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""""设定词向量训练的参数，开始训练词向量"""
num_features = 300      # 词向量取300维
min_word_count = 5     # 词频小于min_word_count个单词就去掉
num_workers = 4         # 并行运行的线程数
context = 10            # 上下文滑动窗口的大小
model_ = 0              # 使用CBOW模型进行训练

model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)

# 构建训练语料库
train_path = '../datas/online_shopping_10_cats.csv'
sentences = getSentence(train_path)
print('Training model...')
model = models.Word2Vec(sentences, workers=num_workers, \
            vector_size=num_features, min_count = min_word_count, \
            window = context, sg=model_)

# 保存模型
# 第一种方法保存的文件不能利用文本编辑器查看，但是保存了训练的全部信息，可以在读取后追加训练
# 后一种方法保存为word2vec文本格式，但是保存时丢失了词汇树等部分信息，不能追加训练

model.save(os.path.join('..', 'models', model_name))

model.wv.save_word2vec_format(os.path.join('..','models','word2vec_txt.txt'),binary = False)