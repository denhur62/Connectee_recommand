import re
import pandas as pd
import numpy as np
def refiner(test_data):
    test_data.drop_duplicates(subset = ['reviews'], inplace=True) # 중복 제거
    test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
    test_data['reviews'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any') # Null 값 제거

def Diary_data_merge():
    dic = ["book", "concern", "daliy", "excercise", "hobby",
       "love", "movie", "pet", "restaurant", "study", "travel"]
    for i in dic:
        if i == 'travel':
            train_data = pd.read_csv(
                'data/{}.csv'.format(i), names=['date', 'title', 'content', 'idx'])
        else:
            train_data = pd.read_csv(
                'data/{}.csv'.format(i), names=['title', 'date', 'content', 'idx'])
        train_data['result'] = train_data['title']+" "+train_data['content']
        qe = train_data[['result']]
        if i == 'book':
            pf = qe
        else:
            pf = pd.concat([pf, qe])

    pf.drop_duplicates(subset=['content'], inplace=True)
    print(pf.isnull().sum())
    # content null값 제거
    pf = pf.dropna(how='any')
    pf['content'] = pf['content'].str.replace(
        "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    pf['content'] = pf['content'].str.replace(
        '^ +', "")  # white space 데이터를 empty value로 변경
    pf['content'].replace('', np.nan, inplace=True)
    pf = pf.dropna(how='any')
    pf.to_csv("diarydata.csv", index=False)