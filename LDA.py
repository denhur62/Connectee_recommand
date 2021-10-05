from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
from gensim import matutils
from ast import literal_eval
from model.database import db_execute
from dataloader import init_vocab_read, tokenizer
import os
import platform
import multiprocessing as mp
import operator
# HyperParameter
NUM_TOPICS = 10
PASSES = 50
ITERATION = 40
MIN_COUNT = 30
# 경로설정
os_platform = platform.platform()
if os_platform.startswith("Windows"):
    model_path = os.getcwd()+'\\model\\lda\\LDA'
    dict_path = os.getcwd()+'\\model\\lda\\dict\\dict'
else:
    model_path = os.getcwd()+'/model/lda/LDA'
    dict_path = os.getcwd()+'/model/lda/dict/dict'

default_dict = dictionary = Dictionary.load(dict_path)
default_model = model = LdaModel.load(datapath(model_path))
# 모델 저장하기


def save_model(model, dictionary, model_path=model_path, dict_path=dict_path):
    model.save(datapath(model_path))
    dictionary.save(dict_path)
    print("LDA model saved")

# 모델 불러오기


def load_model(model_path=model_path, dict_path=dict_path):
    dictionary = Dictionary.load(dict_path)
    model = LdaModel.load(datapath(model_path))
    print("LDA model loaded")
    return model, dictionary


# vec 유사도
def vec_sim(vec_A, vec_B):
    return matutils.cossim(vec_A, vec_B)
# corpus 생성


def make_corpus(model=default_model, dictionary=default_dict):
    sql = "select title,content from diaries where train=0"
    res = db_execute(sql)

    corpus_list = []
    if not res:
        print("there's no make LDA corpus...")
        return '', ''
    else:
        print("make LDA corpus...")
        for doc in res:
            content = doc['title']+" "+doc['content']
            corpus = tokenizer(content)
            corpus_list.append(corpus)
        dictionary.add_documents(corpus_list)
        corpus = [dictionary.doc2bow(d) for d in corpus_list]
        sql = "update diaries set train=1 where train=0"
        db_execute(sql)
        dictionary.filter_extremes(no_below=MIN_COUNT)
        return corpus, dictionary

# 다이어리에 토픽 추가


def insert_topic(diary_id, title, content, model=default_model):
    source = title + " " + content
    corpus = tokenizer(source)
    corpus = [dictionary.doc2bow(d) for d in [corpus]]
    topic = model.get_document_topics(
        bow=corpus[0], minimum_probability=0, per_word_topics=True)[-1]
    if not topic or len(topic) == 1:
        pass
    else:
        stopic = {}
        for i in topic:
            hap = 0
            for j in i[1]:
                hap += j[1]
            stopic[i[0]] = hap
            sdict = sorted(stopic.items(),
                           key=operator.itemgetter(1), reverse=True)
        # 최상위 2개
        interest = dictionary[sdict[0][0]]+','+dictionary[sdict[1][0]]
        sql = "update diaries set interest=%s where id=%s"
        db_execute(sql, (interest, diary_id))

# 다이어리 백터 추가


def insert_diary_vec(diary_id, title, content, model=default_model, dictionary=default_dict):
    source = title + " " + content
    corpus = [dictionary.doc2bow(d)
              for d in [tokenizer(source)]]
    if not corpus:
        vector = []
    else:
        vector = model[corpus[0]]
    vector = str(vector)
    sql = "update diaries set LDAVector=%s where id=%s"
    db_execute(sql, (vector, diary_id))

# 추천 시스템


def recommand(user_id, model=default_model, dictionary=default_dict):
    sql = "select interest from users where id=%s"
    user_interest = db_execute(sql, [user_id])
    corpus = [dictionary.doc2bow(d)
              for d in [tokenizer(user_interest[0]['interest'])]]
    user_interest = model[corpus[0]]
    sql = "select id,LDAVector from diaries where userId!=%s and deletedAt is null"
    result = db_execute(sql, [user_id])
    similar_res = []
    for doc in result:
        res = {}
        q = literal_eval(doc['LDAVector'])
        res['similar'] = vec_sim(user_interest, q)
        res['id'] = doc['id']
        similar_res.append(res)
    return similar_res


def train(corpus, dictionary, update=False, num_topics=NUM_TOPICS, passes=PASSES,
          iterations=ITERATION, model=default_model):
    print("LDA Training...")
    if update:
        model.update(corpus)
    else:
        data = init_vocab_read()
        dictionary = Dictionary(data)
        corpus = [dictionary.doc2bow(d) for d in data]
        model = LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=passes,
            iterations=iterations
        )
    return model, dictionary

# 감정 클릭시 토픽 변경


def emotion_click(user_id, diary_id):
    sql = "select interest from diaries where id=%s"
    diary_topic = db_execute(sql, [diary_id])[0]['interest']
    sql = "select interest from users where id=%s"
    user_interest = db_execute(sql, [user_id])[0]['interest']
    if diary_topic:
        if not user_interest:
            sql = "update users set interest=%s where id=%s"
            db_execute(sql, (diary_topic, user_id))
        else:
            user_interest = user_interest.split(',')
            len_user = len(user_interest)
            if len_user >= 4:
                user_interest.pop(0)
                if len_user == 5:
                    user_interest.pop(0)
                    user_interest += diary_topic.split(',')
                else:
                    user_interest += diary_topic.split(',')
            else:
                user_interest += diary_topic.split(',')
            user_interest = ",".join(user_interest)
            sql = "update users set interest=%s where id=%s"
            db_execute(sql, (user_interest, user_id))
        
    
# 다이어리 보기만 한 경우

def diary_click(user_id, diary_id):
    sql = "select interest from diaries where id=%s"
    diary_topic = db_execute(sql, [diary_id])
    if diary_topic and diary_topic[0]['interest']:
        print(diary_topic)
        diary_topic = diary_topic[0]['interest']
        sql = "select interest from users where id=%s"
        user_interest = db_execute(sql, [user_id])[0]['interest']
        
        if not user_interest:
            sql = "update users set interest=%s where id=%s"
            db_execute(sql, (diary_topic, user_id))
        else:
            user_interest = user_interest.split(',')
            len_user = len(user_interest)
            if len_user == 5:
                user_interest.pop(0)
                user_interest += diary_topic.split(',')
            else:
                user_interest += diary_topic.split(',')
            user_interest = ",".join(user_interest)
            sql = "update users set interest=%s where id=%s"
            db_execute(sql, (user_interest, user_id))
        
# test code
# 초기 학습


def initTrain():
    print("first train")
    data = init_vocab_read()
    dictionary = Dictionary(data)
    corpus = [dictionary.doc2bow(d) for d in data]
    model = LdaModel(corpus=corpus, id2word=dictionary,
                     num_topics=10, random_state=1)
    save_model(model, dictionary)

# 다이어리 안에 백터 넣기


def insert_all_diary_vec(model=default_model):
    sql = "update diaries set train=0 where train=1"
    db_execute(sql)
    sql = "select * from diaries "
    result = db_execute(sql)
    id_list = []
    vector_list = []
    for doc in result:
        content = doc['title']+" "+doc['content']
        corpus = [dictionary.doc2bow(d)
                  for d in [tokenizer(content)]]
        vector_list.append(str(model[corpus[0]]))
        id_list.append(doc['id'])
    sql = "update diaries set LDAVector=%s where id=%s"
    arg = zip(vector_list, id_list)
    db_execute(sql, arg, True)
    sql = "update diaries set train=1 where train=0"
    db_execute(sql)
    print("=======update all diary=========")
# 모델의 모든 토픽 정보 출력


def show_topics(model=default_model, num_words=5):
    topics = model.print_topics(
        num_topics=-1,
        num_words=num_words)
    # 토픽 및 토픽에 대한 단어의 기여도
    for topic in topics:
        print(topic)

