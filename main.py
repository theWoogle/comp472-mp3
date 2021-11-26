import model
import numpy as np
from logger import Logger

if __name__ == "__main__":

    # Download models --> Takes some minutes
    google_300 = model.Model('word2vec-google-news-300')

    #Task2
    #2 new models from different corpora
    twitter_200 = model.Model('glove-twitter-200')
    wiki_200 = model.Model('glove-wiki-gigaword-200')
    #2 new models from the same corpus but different embedding sizes
    twitter_100 = model.Model('glove-twitter-100')
    wiki_100 = model.Model('glove-wiki-gigaword-100')


    # Download datasets
    synonyms = np.loadtxt('data/synonyms.csv', dtype=str, delimiter=',', skiprows=1)

    # Evaluate model 1 word2vec-google-news-300
    google_300.evaluate(synonyms, 4)

    #Task2
    # Evaluate model 2  glove-twitter-200
    twitter_200.evaluate(synonyms,4)
    # Evaluate model 3 glove-wiki-gigaword-200
    wiki_200.evaluate(synonyms,4)
    # Evaluate model 4 glove-twitter-100
    twitter_100.evaluate(synonyms,4)
    # Evaluate model 5 glove-wiki-gigaword-100
    wiki_100.evaluate(synonyms,4)

    # Plot results