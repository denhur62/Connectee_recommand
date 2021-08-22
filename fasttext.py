from gensim.models import FastText
import numpy as np
from Tokenizer import init_vocab_read
from numpy.linalg import norm
from gensim import matutils
import MySQLdb
from Tokenizer import tokenizer
from database import db_execute
import base64
# HyperParameter
# 벡터 차원 수
#VEC_SIZE = 30 (희값)
VEC_SIZE = 100
# 연관 지을 윈도우 사이즈
WINDOWS = 10
# 최소 등장 횟수로 제한
MIN_COUNT = 30
# 모델 에포크
ITERATION = 1000
# 병렬처리 워커수
WORKERS = 16
# vector float 값은 32
model_path = 'model/fasttext/fasttext'
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

# 두 벡터 간의 코사인 유사도 측정


def vec_sim(vec_A, vec_B, model=default_model):
    return np.dot(vec_A, vec_B)/(norm(vec_A)*norm(vec_B))

# 해당 단어 리스트의 벡터값 추출


def get_doc_vector(doc, model=default_model):
    v = [model.wv[word] for word in doc]
    return matutils.unitvec(np.array(v).mean(axis=0))

# Fasttext 추천


def recommand(user_id):
    sql = "select interest from users where id=%s"
    user_interest = db_execute(sql, [user_id])
    user_interest = get_doc_vector(tokenizer(user_interest[0]['interest']))
    print(user_interest.shape)
    sql = "select id,vector from diaries"
    result = db_execute(sql)
    similar_res = []
    for doc in result:
        from ast import literal_eval
        res = {}
        # source = base64.decodestring(doc['vector'])
        # q = np.frombuffer(doc['vector'].encode(), dtype=np.float32)
        q = np.fromstring(doc['vector'][1:-1], dtype=np.float32, sep=' ')
        res['similar'] = vec_sim(user_interest, q)
        res['id'] = doc['id']
        similar_res.append(res)
    print(similar_res)


def insert_all_diary_vec():
    sql = "select * from diaries"
    result = db_execute(sql)
    id_list = []
    vector_list = []
    for doc in result:
        content = doc['title']+" "+doc['content']
        source = get_doc_vector(tokenizer(content))
        # source = base64.b64encode(source)
        vector_list.append(source)
        id_list.append(doc['id'])
    sql = "update diaries set vector=%s where id=%s"
    arg = zip(vector_list, id_list)
    db_execute(sql, arg, True)
    sql = "update diaries set train=1 where train=0"
    db_execute(sql)
    print("=======update all diary=========")


def insert_diary_vec(diary_id, title, content):
    source = title + " " + content
    vector = get_doc_vector(tokenizer(source))
    vector = str(vector)
    sql = "update diaries set vector=%s where id=%s"
    db_execute(sql, (vector, diary_id))


def first_train():
    words = init_vocab_read()
    model = FastText(words, window=1, min_count=2, workers=4, sg=1)
    model.build_vocab(words)

    model.save(model_path)
    model.train(words, total_examples=len(words), epochs=10)


def insert_diary_vec_example(doc):
    tokens = tokenizer(doc)
    vector = get_doc_vector(tokens)
    qw = vector.tobytes()
    vector = np.frombuffer(qw, dtype=np.float32)
    print(vector.dtype)
    # add connection


# first_train()
# insert_diary_vec(
#     2, "오늘은 즐거운 하루", "오늘 길가다가 귀여운 고양이를 봐서 기분이 좋아졌다. 앞으로도 이런 행복한 이들이 나에게도 있었으면 좋겠다 앞으로도 쭈욱욱 아 행복해 일기 일상 여행")
insert_all_diary_vec()
recommand(1)
