import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch
import jieba
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from build_sentence_vector import build_sentence_vec
print('loading all-in-one')
all_in_one=pd.read_csv('/Users/jasondennis/Desktop/graduate-design/data/all-in-one.csv')
# print('load positive')
# pos=pd.read_csv('/Users/jasondennis/Desktop/graduate-design/data/positive.csv')
# print('load negative')
# neg=pd.read_csv('/Users/jasondennis/Desktop/graduate-design/data/negative.csv')
# print(len(all_in_one))
# print('str-ing')
# all_word=str(all_in_one)
# print('done')
# print(len(all_word))
# print('jieba_cuting')
# all_in_one['words']=jieba.cut(all_in_one['Comment'])
# pos['words']=jieba.cut(pos['Comment'])
# neg['words']=jieba.cut(neg['Comment'])
# print('jieba_cut_done')


#这里使用开源知乎word2vec预训练词向量
print('loading word2vec model')
w2v_model=KeyedVectors.load_word2vec_format("sgns.zhihu.bigram.bz2",binary=False)
print('loading done')
#w2v_model = gensim.models.Word2Vec.load('/Users/jasondennis/Desktop/graduate-design/CNN/test1/word_seg_vectors_arr.pkl')

#生成句向量
print('build_sentence_vector')
sentence=[ ]
score=[ ]
for x in range(len(all_in_one)):
    sentence.append(build_sentence_vec(all_in_one['Comment'][x],300,w2v_model))
    score.append(all_in_one['posneg'][x])
#sentence_vec=build_sentence_vec(all_in_one['Comment'],300,w2v_model)
print('done')












