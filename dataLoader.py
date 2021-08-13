import pandas as pd

def data_loader(category):
    data = pd.read_csv('data/{}.csv'.format(category),names=['title', 'date', 'content', 'idx'])
    data = data['content']
    return list(data)[1:]