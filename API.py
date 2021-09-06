import Fasttext
import LDA


def recommand_diary(user_id):
    FT_recommand = Fasttext.recommand(user_id)
    LDA_recommand = LDA.recommand(user_id)
    result = []
    for i in range(len(FT_recommand)):
        temp = {}
        temp['id'] = FT_recommand[i]['id']
        temp['score'] = FT_recommand[i]['similar']+LDA_recommand[i]['similar']
        result.append(temp)
    result.sort(key=lambda x: x['score'], reverse=True)
    temp = [i['id'] for i in result]
    return temp


def vector(diary_id, title, content):
    Fasttext.insert_diary_vec(diary_id, title, content)
    LDA.insert_diary_vec(diary_id, title, content)
    LDA.insert_topic(diary_id, title, content)


def click_emotion(user_id, diary_id):
    LDA.emotion_click(user_id, diary_id)


def click_diary(user_id, diary_id):
    LDA.diary_click(user_id, diary_id)


def search_group(search):
    return Fasttext.group_recommand(search)
