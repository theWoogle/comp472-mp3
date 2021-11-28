import model
import numpy as np
from logger import Logger
import utils
import numpy as np

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
    accuracy = google_300.evaluate(synonyms, 4)
    models_accuracy = np.array(accuracy)
    #Task2
    # Evaluate model 2  glove-twitter-200
    accuracy = twitter_200.evaluate(synonyms,4)
    models_accuracy = np.append(models_accuracy, accuracy)
    # Evaluate model 3 glove-wiki-gigaword-200
    accuracy = wiki_200.evaluate(synonyms,4)
    models_accuracy = np.append(models_accuracy, accuracy)
    # Evaluate model 4 glove-twitter-100
    accuracy = twitter_100.evaluate(synonyms,4)
    models_accuracy = np.append(models_accuracy, accuracy)
    # Evaluate model 5 glove-wiki-gigaword-100
    accuracy = wiki_100.evaluate(synonyms,4)
    models_accuracy = np.append(models_accuracy, accuracy)

    # Plot results
    models=np.array(['google_300','twitter-200','wiki_200', 'twitter_100','wiki_100'])
    print(models)
    print(models_accuracy)
    utils.Plotter(models,models_accuracy)