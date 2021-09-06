from gensim.models.coherencemodel import CoherenceModel
from gensim import models
import pyLDAvis.gensim
from gensim.models.ldamulticore import LdaMulticore
import matplotlib.pyplot as plt


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                         num_topics=num_topics)
        print("###"+str(start)+"###")
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def find_optimal_number_of_topics(dictionary, corpus, processed_data):
    limit = 50  # 토픽 마지막갯수
    start = 2  # 토픽 시작갯수
    step = 6
    model_list, coherence_values = compute_coherence_values(
        dictionary=dictionary, corpus=corpus, texts=processed_data, start=start, limit=limit, step=step)
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

# 모델 시각화


def visualization(ldamodel, corpus, dictionary, name=""):
    print("graphing...")
    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    print("downloading...")
    pyLDAvis.save_html(vis, name + "gensim_output.html")
    # print("displaying...")
    # pyLDAvis.show(vis)


# example
# data = init_vocab_read()
# dictionary = corpora.Dictionary(data)
# corpus = [dictionary.doc2bow(d) for d in data]
# find_optimal_number_of_topics(dictionary, corpus, data)
