from gensim.models import Word2Vec

from utils import get_config

if __name__ == '__main__':
    config, unparsed = get_config(return_unparsed=True)

    if len(unparsed) != 1:
        print('Please provided a word')
        exit(1)

    word = unparsed[0]

    model = Word2Vec.load(config.embeddings_model)
    print('Words similar to ' + word)
    for w, s in model.most_similar(word):
        print('{:>15} {:.2f}%'.format(w, s * 100))
