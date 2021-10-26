from konlpy.tag import Mecab
import pandas as pd
import pickle
import platform
from torch.utils.data import Dataset
import torch
from model.database import db_execute
stopwords = pd.read_csv('token/stopwords.csv')
stopwords = list(stopwords['stopword'])
os_platform = platform.platform()


def tokenizer(raw, pos=["NNG", "NNP", "VV", "VA"], stopword=stopwords):
    if os_platform.startswith("Windows"):
        m = Mecab("C:/mecab/mecab-ko-dic")
    else:
        m = Mecab()
    return [word for word, tag in m.pos(raw) if len(word) > 1 and tag in pos and word not in stopword]


class VAEDataset(Dataset):

    # 데이터 정의
    def __init__(self, x_data, y_data=None):
        self.x_data = x_data
        self.y_data = y_data

    # 이 데이터 셋의 총 데이터 수
    def __len__(self):
        return len(self.x_data)

    # 어떠한 idx를 받았을 때 그에 맞는 데이터를 반환
    def __getitem__(self, idx):
        if self.y_data is None:
            x = torch.FloatTensor(self.x_data[idx])
            return x
        else:
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y

def get_click_comment():
    # clicks frame
    sql = """select * from clicks where deletedAt is null"""
    re = db_execute(sql)
    cf = pd.DataFrame(re)
    cf['createdAt'] = cf['createdAt'].dt.strftime('%y-%m-%d')
    cf['one'] = 1
    cf = cf.groupby(['userId', 'createdAt', 
    'diaryId', 'emotionType', 'emotionLevel'])['one'].sum().reset_index()
    cf_emotion = pd.pivot_table(cf, index=['userId', 'createdAt'],
                                columns=['emotionType'], values='emotionLevel')
    cf_diary = pd.pivot_table(cf, index=['userId', 'createdAt'],
                              columns=['diaryId'], values='one')
    cf = pd.merge(cf_emotion, cf_diary, 'left', on=['userId', 'createdAt'])

    # comments frame
    sql = "select * from comments where deletedAt is null"
    result = db_execute(sql)
    dq = pd.DataFrame(result)
    dq['createdAt'] = dq['updatedAt'].dt.strftime('%y-%m-%d')
    dq_emotion = pd.pivot_table(dq, index=['userId', 'createdAt'],
                                columns=['userEmotionType'], values='userEmotionLevel')
    dq_diary = pd.pivot_table(dq, index=['userId', 'createdAt'],
                              columns=['diaryId'], values='emotionLevel')
    dq = pd.merge(dq_emotion, dq_diary, 'left', on=['userId', 'createdAt'])
    da = pd.concat([cf, dq]).fillna(0).sort_index()
    return da

# 배포후 정리 
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
