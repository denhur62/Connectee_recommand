from konlpy.tag import Mecab
import pandas as pd
import pickle

stopwords=pd.read_csv('stopwords.csv')

stopwords =list(stopwords['stopword'])

def tokenizer(raw, pos=["NNG", "NNP", "VV", "VA"], stopword=stopwords):
    m = Mecab()
    q=[]
    for word,tag in m.pos(raw):
        if len(word) > 1 and tag in pos and word not in stopword:
            q.append(word)
    return q
    # return [word for word, tag in m.pos(raw) if len(word) > 1 and tag in pos and word not in stopword]

qw = pd.read_csv('data2.csv')
data_list=[]
data=qw['content']
with open('./vocab.pkl', 'wb') as f:
    for i in data:
        pickle.dump(tokenizer(i), f)

with open('./vocab.pkl', 'rb') as f:
    while True:
        try:
            data = pickle.load(f, encoding = 'utf-8')
        except EOFError:
            break
        data_list.append(data)
print(len(data_list))