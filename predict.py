import Fasttext
import LDA
import VAE
from collections import defaultdict 
from model.database import db_execute

def find_depression(user_id):
    sql = "select emotionType from comments where userId=%s order by updatedAt desc"
    user_emotion = db_execute(sql, [user_id])
    if not user_emotion:
        return False
    for i in user_emotion[:3]:
        if (i['emotionType']!='sad'  and i['emotionType']!='disgusted' 
        and i['emotionType']!='terrified'):
            return False
    return True

def recommand_diary(user_id):
    if find_depression(user_id):
        sql ="select diaryId from comments order by field(emotionType, 'surprised','happy') desc"
        result = db_execute(sql)
        temp = [i['diaryId'] for i in result]
        return temp

    FT_recommand = Fasttext.recommand(user_id)
    LDA_recommand = LDA.recommand(user_id)
    VAE_recommand = defaultdict(int,VAE.recommand(user_id))
    result = []
    for i in range(len(FT_recommand)):
        temp = {}
        temp['id'] = FT_recommand[i]['id']
        temp['score'] = FT_recommand[i]['similar']+LDA_recommand[i]['similar']+ \
            VAE_recommand[temp['id']]
        result.append(temp)
    result.sort(key=lambda x: x['score'], reverse=True)
    temp = [i['id'] for i in result]
    return temp
    
def insert_vector(diary_id, title, content):
    Fasttext.insert_diary_vec(diary_id, title, content)
    LDA.insert_diary_vec(diary_id, title, content)
    LDA.insert_topic(diary_id, title, content)

