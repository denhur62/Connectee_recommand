import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc, rcParams
from gensim.models import FastText
# font settings
font_name = font_manager.FontProperties(
    fname=r"c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
rcParams.update({'figure.autolayout': True})
model_path = '../model/fasttext'
ko_model = FastText.load(model_path)
# pip uninstall matplotlib && pip uninstall sklearn

# Limit number of tokens to be visualized
limit = 5000
vector_dim = 100

# Getting tokens and vectors
words = []
embedding = np.array([])
i = 0
# wv.vocab -> wv.key_to_index
for word in ko_model.wv.key_to_index:
    # Break the loop if limit exceeds
    if i == limit:
        break

    # Getting token
    words.append(word)

    # Appending the vectors
    embedding = np.append(embedding, ko_model.wv.get_vector(word, norm=True))
    i += 1

# Reshaping the embedding vector
print(embedding.shape)
embedding = embedding.reshape(limit, vector_dim)


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(10, 4),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)


# Creating the tsne plot [Warning: will take time]
tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)

low_dim_embedding = tsne.fit_transform(embedding)

# Finally plotting and saving the fig
plot_with_labels(low_dim_embedding, words)
