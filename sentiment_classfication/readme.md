## 一、word2vector模型训练生成词向量

### 1、运行文件

文件：word2vector.py

### 2、代码说明：

#### (1) 日志打印

```python
"""打印日志信息"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

这里完成了日志系统的基本配置。

level：将根记录器级别设置为指定的级别。级别排序：CRITICAL > ERROR > WARNING > INFO > DEBUG。（这里设置成INFO，高于这个级别的日志才会被输出。）


#### （2）参数设置

```python
""""设定词向量训练的参数，开始训练词向量"""
num_features = 300      # 词向量取300维
min_word_count = 5      # 词频小于min_word_count个单词就去掉
num_workers = 4         # 并行运行的线程数
context = 10            # 上下文滑动窗口的大小
model_ = 0              # 使用CBOW模型进行训练
```



#### (3) 构建训练语料

```python
train_path = '../datas/online_shopping_10_cats.csv'
sentences = getSentence(train_path)
```

函数功能是为了构造训练语料，

原始语料中的每一句形式如下：

```
'作者在少年时即喜阅读，能看出他精读了无数经典，因而他有一个庞大的内心世界。他的作品最难能可贵的有两点，一是他的理科知识不错，虽不能媲及罗素，但与理科知识很差的作家相比，他的文章可读性要强；其二是他人格和文风的朴实，不造作，不买弄，让人喜欢。读他的作品，犹如听一个好友和你谈心，常常唤起心中的强烈的共鸣。他的作品90年后的更好些。衷心祝愿周国平健康快乐，为世人写出更多好作品。'
```

需要对其进行分割，并去除中文停用词：

```
['作者', '在', '少年', '时即', '喜', '阅读', '能', '看出', '精读', '了', '无数', '经典', '因而', '有', '一个', '庞大', '内心世界', '作品', '最', '难能可贵', '有', '两点', '一是', '理科', '知识', '不错', '虽', '不能', '媲及', '罗素', '但', '与', '理科', '知识', '很差', '作家', '相比', '文章', '可读性', '要强', '；', '其二', '是', '人格', '和', '文风', '朴实', '不', '造作', '不买', '弄', '让', '人', '喜欢', '读', '作品', '犹如', '听', '一个', '好友', '和', '谈心', '常常', '唤起', '心中', '强烈', '共鸣', '作品', '90', '年', '后', '更好', '些', '衷心祝愿', '周国平', '健康', '快乐', '为', '世人', '写出', '更', '多', '好', '作品']
```



#### （4）模型训练

代码：

```python
from gensim import models
model = models.Word2Vec(sentences, workers=num_workers, \
            vector_size=num_features, min_count = min_word_count, \
            window = context, sg=model_)
```

这里使用gensim库中的models.word2Vec。

【关于gensim】Gensim是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。它支持包括TF-IDF，LSA，LDA，和word2vec在内的多种主题模型算法，
支持流式训练，并提供了诸如相似度计算，信息检索等一些常用任务的API接口。（参考：[15分钟入门Gensim - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/37175253)）



#### （5）训练完成后模型保存

```python
# 保存模型
# 第一种方法保存的文件不能利用文本编辑器查看，但是保存了训练的全部信息，可以在读取后追加训练
# 后一种方法保存为word2vec文本格式，但是保存时丢失了词汇树等部分信息，不能追加训练

model.save(os.path.join('..', 'models', model_name))

model.wv.save_word2vec_format(os.path.join('..','models','word2vec_txt.txt'),binary = False)
```

保存到了./models文件夹下，两种保存方式对应的是这个文件夹下的两个文件。

word2vec_txt.txt可以直接打开，可以查看每个分词的向量。



## 二、使用bilstm模型进行情感分类

### 1、文件位置

bilstm模型:`bilstm-test/model.py`



### 2、模型说明

#### 1、两种模式：

（1）带attention的bilstm。

（2）不带attention的bilstm。

注意：训练时默认关闭attention。

#### 2、损失函数：交叉熵损失函数

参考：[CrossEntropyLoss — PyTorch 1.11.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

## 三、使用n-gram语言模型与随机森林分类器进行分类
### 1、文件位置：n-gram
