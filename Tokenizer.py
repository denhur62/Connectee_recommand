from konlpy.tag import Mecab
import pandas as pd
import pickle

stopwords = pd.read_csv('token/stopwords.csv')
stopwords = list(stopwords['stopword'])


def tokenizer(raw, pos=["NNG", "NNP", "VV", "VA"], stopword=stopwords):
    # linux
    m = Mecab()
    # window
    #m = Mecab("C:/mecab/mecab-ko-dic")
    return [word for word, tag in m.pos(raw) if len(word) > 1 and tag in pos and word not in stopword]


def init_vocab_write():
    data = pd.read_csv('data2.csv')
    data = data['content']
    with open('model/vocab.pkl', 'wb') as f:
        for i in data:
            pickle.dump(tokenizer(i), f)


def init_vocab_read():
    data_list = []
    with open('model/vocab.pkl', 'rb') as f:
        while True:
            try:
                data = pickle.load(f, encoding='utf-8')
            except EOFError:
                break
            data_list.append(data)
    return data_list
