import pandas as pd
import numpy as np
dic = ["book", "concern", "daliy", "excercise", "hobby",
       "love", "movie", "pet", "restaurant", "study", "travel"]
# for i in dic:
#     if i == 'travel':
#         train_data = pd.read_csv(
#             'data/{}.csv'.format(i), names=['date', 'title', 'content', 'idx'])
#     else:
#         train_data = pd.read_csv(
#             'data/{}.csv'.format(i), names=['title', 'date', 'content', 'idx'])
#     train_data['result'] = train_data['title']+" "+train_data['content']
#     qe = train_data[['result']]
#     if i == 'book':
#         pf = qe
#         print(12)
#     else:
#         pf = pd.concat([pf, qe])
# pf.to_csv("data/data.csv", index=False)
train_data = pd.read_csv(
    'data/data.csv', names=['content'])
train_data.drop_duplicates(subset=['content'], inplace=True)
print(train_data.isnull().sum())
# content null값 제거
train_data = train_data.dropna(how='any')
train_data['content'] = train_data['content'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data['content'] = train_data['content'].str.replace(
    '^ +', "")  # white space 데이터를 empty value로 변경
train_data['content'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')
train_data.to_csv("data/data2.csv", index=False)
