from gensim import matutils
from gensim.models import FastText
import numpy as np
from numpy.linalg import norm
from dataloader import tokenizer, init_vocab_read
from model.database import db_execute
import csv
# vector float 값은 32
# HyperParameter
# 벡터 차원 수
VEC_SIZE = 30
WINDOWS = 10
MIN_COUNT = 30
EPOCH = 100
WORKERS = 16

model_path = 'model/fasttext/fasttext'
META_FILE_TSV = 'token/words.tsv'
default_model = FastText.load(model_path)
# 모델 저장


def save_model(model, path=model_path):
    model.save(path)
    print("Fasttext model saved")

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
    user_interest = db_execute(sql, [user_id])[0]['interest']
    if not user_interest:
        user_interest = []
    else:
        user_interest = get_doc_vector(tokenizer(user_interest))
    sql = """select id,FTVector 
    from diaries where userId!=%s and deletedAt is null
    and private=0"""
    result = db_execute(sql, [user_id])
    similar_res = []
    for doc in result:
        res = {}
        if len(doc['FTVector']) <= 2:
            res['similar'] = -1
        elif type(user_interest) == list:
            res['similar'] = -1
        else:
            q = np.fromstring(doc['FTVector'][1:-1], dtype=np.float32, sep=' ')
            res['similar'] = vec_sim(user_interest, q)
        res['id'] = doc['id']
        similar_res.append(res)
    return similar_res

# 그룹 추천


def group_recommand(search=None):
    # 토크나이저 못할 경우 유사한 그룹만 추천
    if not tokenizer(search):
        serach_like = "%{}%".format(search)
        sql = "select a.id from Connectee.groups as a where a.title like %s"
        re = db_execute(sql, [serach_like])
        if not re:
            temp = []
            return temp
        else:
            temp = [i['id'] for i in re]
            return temp
    search = get_doc_vector(tokenizer(search))
    themes = {1: "취미", 2: "여행", 3: "공부", 4: "운동", 5: "맛집",
              6: "영화", 7: "사랑", 8: "책", 9: "애완동물", 10: "고민"}
    sql = """select a.id,a.title, a.description
    from Connectee.groups as a where deletedAt is null"""
    re = db_execute(sql)
    result = []
    for i in re:
        dic = {}
        dic['id'] = i['id']
        source = i['title'] + " " + i['description']
        temp = ''
        sql = "select GroupId,ThemeId from group_themes where GroupId=%s and deletedAt is null"
        theme = db_execute(sql, [dic['id']])
        if not theme:
            pass
        else:
            for j in theme:
                temp += ' '+themes[j['ThemeId']]
            source += temp
        if not tokenizer(source):
            dic['similar'] = -1
        else:
            vector = get_doc_vector(tokenizer(source))
            if len(vector) <= 2 or type(vector) == list:
                dic['similar'] = -1
            else:
                dic['similar'] = vec_sim(search, vector)
        result.append(dic)
    result.sort(key=lambda x: x['similar'], reverse=True)
    temp = [i['id'] for i in result]
    return temp


# 다이어리 입력후 벡터 저장


def insert_diary_vec(diary_id, title, content):
    source = title + " " + content
    content = tokenizer(source)
    if not content:
        vector = []
    else:
        vector = get_doc_vector(content)
    vector = str(vector)
    sql = "update diaries set FTVector=%s where id=%s"
    db_execute(sql, (vector, diary_id))

# 추가 학습 단어 corpus 추가 테스트


def make_corpus(model=default_model):
    sql = "select title,content from diaries where train=0"
    res = db_execute(sql)

    corpus_list = []
    if not res:
        print("there's no make FT corpus...")
        pass
    else:
        print("make FT corpus...")
        for doc in res:
            content = doc['title']+" "+doc['content']
            corpus = tokenizer(content)
            corpus_list.append(corpus)
    return corpus_list


# 단어 저장
def make_tsv(model=default_model, meta_file_tsv=META_FILE_TSV):
    print("make tsv...")
    with open(meta_file_tsv, 'w', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        words = list(model.wv.index_to_key)
        for word in words:
            writer.writerow([word])


def train(corpus, update=False, model=default_model, vec_size=VEC_SIZE, windows=WINDOWS, min_count=MIN_COUNT, epochs=EPOCH,
          workers=WORKERS):
    print("Training...")
    if update:
        model.build_vocab(corpus, update=update)
        model.train(corpus, total_examples=len(corpus), epochs=model.epochs)
    else:
        corpus = init_vocab_read()
        model = FastText(vector_size=vec_size,
                         window=windows,
                         min_count=min_count,
                         sentences=corpus,
                         epochs=EPOCH,
                         workers=workers)
    return model

