import math
import numpy as np

def build_sentence_vec(sentence,size,w2v_model):
    sen_vec=np.zeros(size).reshape((1,size))
    count=0
    for word in sentence:
        try:
            sen_vec+=w2v_model[word].reshape((1,size))
            count+=1
        except KeyError:
            continue
    if count!=0:
        sen_vec/=count
    return sen_vec


def build_sentence_vec_weight(sentence,size,w2v_model,key_weight):
    key_words_list=list(key_weight)
    sen_vec=np.zeros(size).reshape((1,size))
    count=0
    for word in sentence:
        try:
            if word in key_words_list:
                sen_vec+=(np.dot(w2v_model[word],math.exp(key_weight[word]))).reshape((1,size))
                count+=1
            else:
                sen_vec+=w2v_model[word].reshape((1,size))
                count+=1
        except KeyError:
            continue
    if count!=0:
        sen_vec/=count
    return sen_vec