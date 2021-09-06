import Fasttext
import LDA


def FTtrain():
    print("#----FastText----#")
    corpus = Fasttext.make_corpus()
    if corpus:
        model = Fasttext.train(corpus, True)
        Fasttext.save_model(model)
        Fasttext.make_tsv(model)


def LDAtrain():
    print("#----LDA----#")
    corpus, dictionary = LDA.make_corpus()
    if corpus:
        model, dictionary = LDA.train(corpus, dictionary, True)
        LDA.save_model(model, dictionary)
