from gensim.models import FastText
import numpy as np
from Tokenizer import init_vocab_read
from gensim import matutils
# HyperParameter
# 벡터 차원 수
VEC_SIZE = 30
# 연관 지을 윈도우 사이즈
WINDOWS = 10
# 최소 등장 횟수로 제한
MIN_COUNT = 30
# 모델 에포크
ITERATION = 1000
# 병렬처리 워커수
WORKERS = 16

model_path = 'model/fasttext'
default_model = FastText.load(model_path)

# 모델 저장


def model_save(model, path=model_path):
    model.save(path)

# 모델 로드


def model_load(path=model_path):
    return FastText.load(path)

# 두 단어 리스트 사이의 유사도 측정


def doc_sim(doc_A, doc_B, model=default_model):
    return model.wv.n_similarity(doc_A, doc_B)

# 두 벡터 간의 유사도 측정


def vec_sim(vec_A, vec_B, model=default_model):
    return np.dot(vec_A, vec_B)

# 해당 단어 리스트의 벡터값 추출


def get_doc_vector(doc, model=default_model):
    v = [model.wv[word] for word in doc]
    return matutils.unitvec(np.array(v).mean(axis=0))


def first_train():
    words = init_vocab_read()
    model = FastText(words, window=1, min_count=2, workers=4, sg=1)
    model.build_vocab(words)
    model.save(model_path)
    #model.train(words, total_examples=len(words), epochs=10)


loaded_model = FastText.load(model_path)
print(loaded_model.wv.vectors.shape)
diarys = init_vocab_read()
example = diarys[11]
example_vector = get_doc_vector(example)
diarys = diarys[:10]
diary_list = []
sim_list = []
for diary in diarys:
    diary_list.append(vec_sim(example_vector, get_doc_vector(diary)))
    sim_list.append(doc_sim(example, diary))
print(diary_list)
print(sim_list)

# print(loaded_model.wv.most_similar("일기를 가서 기분이 좋아요 앞으로도 계속", topn=5))
# print(loaded_model.wv.most_similar("여행", topn=5))
# print(loaded_model.wv.similarity("기분 좋아", '여행'))
