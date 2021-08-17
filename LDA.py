import numpy as np
import pandas as pd
import gensim
from gensim.test.utils import datapath
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from Tokenizer import init_vocab_read
# import pyLDAvis.gensim
import matplotlib.pyplot as plt
import os
import platform
import multiprocessing as mp
from multiprocessing import freeze_support

# HyperParameter
WORKERS = 4
NUM_TOPICS = 30
PASSES = 30
EVERY_POST_LIMIT = 20
NAVER_TITLE_LIMIT = 5
TOTAL_POST_LIMIT = 5
ITERATION = 70
MIN_COUNT = 30

os_platform = platform.platform()
model_path = os.getcwd()+'\\model\\LDA'
dict_path = os.getcwd()+'\\dict\\LDA'
# default_dict = corpora.Dictionary.load(dict_path)
# default_lda = gensim.models.ldamodel.LdaModel.load(datapath(model_path))

# 모델 저장하기


def save_model(model, dictionary, model_path=model_path, dict_path=dict_path):
    model.save(datapath(model_path))
    dictionary.save(dict_path)
    print("model saved")

# 모델 불러오기


def load_model(model_path=model_path, dict_path=dict_path):
    dictionary = corpora.Dictionary.load(dict_path)
    lda = gensim.models.ldamodel.LdaModel.load(datapath(model_path))
    print("loaded")
    return lda, dictionary

# 모델의 모든 토픽 정보 출력


# def show_topics(model=default_lda, num_words=5):
#     topics = model.print_topics(
#         num_topics=-1,
#         num_words=num_words)  # 토픽 단어 제한
#     # 토픽 및 토픽에 대한 단어의 기여도
#     for topic in topics:
#         print(topic)
#     return topics

# # 하나의 문서에 대하여 토픽 정보 예측


# def get_topics(doc, model=default_lda, dictionary=default_dict):
#     df = pd.DataFrame({'text': [doc]})
#     if str(type(doc)) == "<class 'list'>":
#         tokenized_doc = df['text']
#     else:
#         tokenized_doc = df['text'].apply(lambda x: get_tk(x))
#     corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
#     for topic_list in model[corpus]:
#         temp = topic_list
#         temp = sorted(topic_list, key=lambda x: (x[1]), reverse=True)
#         break
#     result = np.zeros(NUM_TOPICS)
#     for idx, data in temp:
#         result[idx] += data
#     return result

# # 해당 단어리스트가 딕셔너리에 내에 포함된 단어인지 검증


# def is_valid_words(word_list, dict=default_dict):
#     temp = dict.doc2idx(word_list)
#     result = []
#     for i in temp:
#         if i == -1:
#             result += [False]
#         else:
#             result += [True]
#     return result


def initTrain():
    data = init_vocab_read()
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(d) for d in data]
    model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                     num_topics=9, random_state=1)
    save_model(model, dictionary)
    for t in model.show_topics():
        print(t)


# initTrain()
model, dictionary = load_model()
for t in model.show_topics():
    print(t)
