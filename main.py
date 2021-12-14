import model
import numpy as np
import sys
import plotter


if __name__ == "__main__":

    is_test = sys.argv[0]

    # Download dataset
    path = 'data/sample_24.csv' if is_test else 'data/synonyms.csv'
    synonyms = np.loadtxt(path, delimiter=',',
                          dtype=str, skiprows=1)

    model_names = ['word2vec-google-news-300',
                   'glove-twitter-200',
                   'glove-wiki-gigaword-200',
                   'glove-twitter-100',
                   'glove-twitter-50']

    models = []
    models_accuracy = []
    for name in model_names:
        models.append(model.Model(name))

    print(f"Evaluating Models")
    for model in models:
        accuracy = model.evaluate(synonyms, 4)
        models_accuracy.append(accuracy)

    models_n = ['google300',
                'twitter200',
                'wiki200',
                'twitter100',
                'twitter50',
                'humangs',
                'randb']

    # Human gold standard from moodle
    humangs_v = 0.855 if is_test == False else 0
    models_accuracy.append(humangs_v)

    # random guessing 1/4
    models_accuracy.append(0.25)

    plotter.Plotter(models_n, models_accuracy)

    print(f"Models: \t{models_n}")
    print(f"accuracies: \t{models_accuracy}")
