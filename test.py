import pandas as pd
from database import db_execute

# 감정 추가해야함


def get_click_comment():
    sql = "select * from clicks"
    cd = pd.DataFrame(db_execute(sql))
    cd['createdAt'] = cd['createdAt'].dt.strftime('%y-%m-%d')
    cd = cd.groupby(['userId', 'createdAt', 'diaryId'])[
        'clickCount'].sum().reset_index()
    cd = pd.pivot_table(cd, index=['userId', 'createdAt'],
                        columns='diaryId', values='clickCount')
    sql = "select * from comments where deletedAt is null"
    result = db_execute(sql)
    dq = pd.DataFrame(result)
    dq['createdAt'] = dq['updatedAt'].dt.strftime('%y-%m-%d')
    dq = dq.set_index(['userId', 'createdAt', 'diaryId'])
    dq = dq[['emotionLevel']]
    dq = pd.pivot_table(dq, index=['userId', 'createdAt'],
                        columns='diaryId', values='emotionLevel')
    return cd, dq


# df, dq = get_click_comment()
# da = pd.merge(df, dq, on=['userId', 'createdAt'],
#               how='outer', suffixes=['', '_new'], sort=True).fillna(0)
# check = False
# cnt = 0
# for i in da.columns:
#     if check:
#         check = False
#         da = da.drop(columns=[i], axis=1)
#         continue
#     if isinstance(i, str):
#         check = True
#         da[i] = da[i]+da[da.columns[cnt+1]]
#     cnt += 1
# print(da)
sql = "select interest from diaries where id=%s"
diary_topic = db_execute(sql, [2])[0]['interest']
print(diary_topic)
