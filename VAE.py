import pandas as pd
from database import db_execute

# movie_paths = 'data/'

# movie = pd.read_csv(movie_paths + "ratings.csv")
# meta = pd.read_csv(movie_paths + 'movies_metadata.csv', low_memory=False)
# meta = meta.rename(columns={'id': 'movieId'})

# movie['movieId'] = movie['movieId'].astype(str)
# meta['movieId'] = meta['movieId'].astype(str)

# movie = pd.merge(movie, meta[['movieId', 'original_title']], on='movieId')
# movie['one'] = 1

sql = "select * from clicks"
result = db_execute(sql)
df = pd.DataFrame(result)
df['createdAt'] = df['createdAt'].dt.strftime('%y-%m-%d')

df = df.groupby(['userId', 'createdAt', 'diaryId'])[
    'clickCount'].sum().reset_index()
df = pd.pivot_table(df, index=['userId', 'createdAt'],
                    columns='diaryId', values='clickCount')
sql = "select * from comments where deletedAt is null"
result = db_execute(sql)
dq = pd.DataFrame(result)
dq['createdAt'] = dq['updatedAt'].dt.strftime('%y-%m-%d')
dq = dq.set_index(['userId', 'createdAt', 'diaryId'])
dq = dq[['emotionLevel']]
dq = pd.pivot_table(dq, index=['userId', 'createdAt'],
                    columns='diaryId', values='emotionLevel')
da = pd.merge(df, dq, on=['userId', 'createdAt'],
              how='outer', sort=True).fillna(0)
print(da)
