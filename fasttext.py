from gensim.models import FastText
import numpy as np
words = [['hello', 'wrod', 'qwe', 'awer', 'awer'],
         ['qwer', 'qwey', 'qwey', 'awety', 'qweryt', 'computer'],
         ['qwer', 'qwey', 'qw', 'hgaet', 'aweg', 'computer']]
model = FastText(words, window=1, min_count=2, workers=4, sg=1)
model.build_vocab(words)
model.train(words, total_examples=len(words), epochs=10)
model2 = FastText(words, window=1, min_count=2, workers=4, sg=1)
model2.build_vocab(words)
print(np.allclose(model.wv['computer'], model2.wv['computer']))
