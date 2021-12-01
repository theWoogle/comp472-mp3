import model
import numpy as np
from logger import Logger
import utils
import numpy as np

if __name__ == "__main__":

    # Download datasets
    synonyms = np.loadtxt('data/synonyms.csv', dtype=str, delimiter=',', skiprows=1)


    model_names = ['word2vec-google-news-300', 'glove-twitter-200', 'glove-wiki-gigaword-200',
                   'glove-twitter-100', 'glove-wiki-gigaword-100']
    models = []
    models_accuracy=[]
    for name in model_names:
        models.append(model.Model(name))
    for model in models:
        accuracy = model.evaluate(synonyms,4)
        models_accuracy.append(np.array(accuracy))

    models_n = ['google300', 'twitter200', 'wiki200', 'twitter100', 'wiki100','humangs','randb']
    models_accuracy.append(0.855)
    models_accuracy.append(0.25)
    utils.Plotter(models_n, models_accuracy)
    print(models_n)
    print(models_accuracy)
