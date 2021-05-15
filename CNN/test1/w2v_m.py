import pandas as pd
import gensim
import time
import pickle
import numpy as np
import jieba
from gensim.models import word2vec

vector_size = 100

#准备工作
def sentence2list(sentence):
    return sentence.strip().split()

start_time = time.time()

#读取训练数据

print("准备数据................\n")
text_train= pd.read_csv('/Users/jasondennis/Desktop/1000000limit.csv',encoding='utf-8',dtype=str)
sentences_train = list(text_train.loc[:, 'Comment'].apply(sentence2list))

# 首先就是这里，cut的输入应该是一个str字符串类型, sentencetrain的数据类型是一个二维数组，而且只有一列
# 改成如下形式: 因为只有一列，所以其句子就是sentence[0]，然后依次取出，放入列表中，结果== [[word， word。。。]]
sentences = [[word[0] for word in jieba.cut(sentence[0]) if word] for sentence in sentences_train]

# 这里的iter就是将列表转换为iterator，因为gensim的训练接受的数据类型就是这样，在3.x的版本也可以接受generator
word2vec.LineSentence(iter(sentences), max_sentence_length=100000, limit=None)  #  这句话放这里没问题

print("准备数据完成!\n")

#训练词向量模型
print("开始训练................\n")
# 在这里转换为iter(sentence)也可以
model = gensim.models.Word2Vec(sentences,min_count=5,epochs=5)
print("训练完成!\n")

#后续保存
print(" 保存训练结果...........\n")
wv = model.wv
# AttributeError: The index2word attribute has been replaced by index_to_key since Gensim 4.0.0.
# 这里写的很明确了，这个api换了个名字 index2word --> index_to_key
vocab_list = wv.index_to_key

word_idx_dict = {}
for idx, word in enumerate(vocab_list):
    word_idx_dict[word] = idx

vectors_arr = wv.vectors
vectors_arr = np.concatenate((np.zeros(vector_size)[np.newaxis, :], vectors_arr), axis=0)#第0位置的vector为'unk'的vector

f_wordidx = open('./word_seg_word_idx_dict.pkl', 'wb')
f_vectors = open('./word_seg_vectors_arr.pkl', 'wb')
pickle.dump(word_idx_dict, f_wordidx)
pickle.dump(vectors_arr, f_vectors)
f_wordidx.close()
f_vectors.close()
print("训练结果已保存到该目录下！\n")

end_time = time.time()
print("耗时：{}s\n".format(end_time - start_time))