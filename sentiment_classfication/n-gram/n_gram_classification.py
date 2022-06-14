from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from util.util import getSentence, model_eval
from sklearn.ensemble import RandomForestClassifier

"""使用sklearn计算n-gram，得到词语-文本矩阵"""
n_gram = 2  #设置n_gram的值
train_path='../datas/online_shopping_10_cats.csv'
df_train = pd.read_csv(train_path, names=['cat', 'label', 'text'], sep=',')

sentences_train = getSentence(train_path)
sentences_train = [' '.join(s) for s in sentences_train]

# n-gram分词，生成对应词语-文本矩阵
vectorizer_2gram = CountVectorizer(ngram_range=(n_gram,n_gram),token_pattern=r'\b\w+\b',max_features=5000)
vectorizer_2gram.fit(sentences_train)
train_vsm_2gram = vectorizer_2gram.transform(sentences_train).toarray()

# print(train_vsm_2gram)
# 训练随机森林分类器
forest = RandomForestClassifier(oob_score=True,n_estimators = 200)
forest = forest.fit(train_vsm_2gram, df_train.cat)

#测试模型
test_path='../datas/test_online_shopping_10_cats.csv'
df_test = pd.read_csv(test_path, names=['cat', 'label', 'text'], sep=',')

sentences_test = getSentence(test_path)
sentences_test = [' '.join(s) for s in sentences_test]

test_vsm_2gram = vectorizer_2gram.transform(sentences_test).toarray()

model_eval(forest, test_vsm_2gram, df_test.cat)
