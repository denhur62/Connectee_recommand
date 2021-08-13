from gensim.models import FastText
import numpy as np
from Tokenizer import tokenizer
import pandas as pd
data = pd.read_csv('data2.csv')
data=data[:5]
data=data['content'].tolist()
data="".join(data)
print(type(data))
words=tokenizer(data)
# model = FastText(words, window=1, min_count=2, workers=4, sg=1)
# model.build_vocab(words)
# model.train(words, total_examples=len(words), epochs=10)


