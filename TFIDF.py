from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PartitionFit import partial_fit
from Tokenizer import tokenizer, stopwords
from DataLoader import data_loader
import _pickle as pickle
import scipy.sparse as sp
TfidfVectorizer.partial_fit = partial_fit
tf = TfidfVectorizer(
    stop_words=stopwords, tokenizer=tokenizer, ngram_range=(1, 2), max_features=200000, sublinear_tf=True)


catergory = ['book', 'concern', 'daliy', 'excercise', 'hobby',
             'love', 'movie', 'pet', 'restaurant', 'study', 'travel']
mtx_path = 'model/model.mtx'
tf_path = 'model/tf.pickle'


def first_learning(tf, mtx_path):
    article = []
    for i in catergory:
        data = data_loader(i)
        article += data
    X = tf.fit_transform(article)
    tf.n_docs = len(article)
    with open(mtx_path, "wb") as fw:
        pickle.dump(X, fw)
    with open(tf_path, "wb") as fw:
        pickle.dump(tf, fw)


def after_learning(tf, more):
    tf.partial_fit([more])


def recommand(more):
    with open(mtx_path, "rb") as fr:
        X = pickle.load(fr)
    with open(tf_path, "rb") as fr:
        tf = pickle.load(fr)
    # X = mmread(mtx_path).tocsr()
    example_vector = tf.transform([more])
    tf.partial_fit([more])
    with open(tf_path, "wb") as fw:
        pickle.dump(tf, fw)

    cos_similar = linear_kernel(example_vector, X).flatten()

    sim_rank_idx = cos_similar.argsort()[::-1]
    sim_rank_idx = sim_rank_idx[:5]
    for i in sim_rank_idx:
        print(cos_similar[i])


first_learning(tf, mtx_path)

recommand('오늘은 책을 읽고 싶은 밤이다. 일기를 쓰려고 누웠는데 책이 눈에 들어왔다. 내돈내산하고 작성하는 책 리뷰 여행을 가고자 하였다.')
