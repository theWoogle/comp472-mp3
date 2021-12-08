import model
import numpy as np
import plotter
import numpy as np

if __name__ == "__main__":

    # Download dataset
    synonyms = np.loadtxt('data/synonyms.csv', dtype=str,
                          delimiter=',', skiprows=1)

    model_names = ['word2vec-google-news-300', 'glove-twitter-200', 'glove-wiki-gigaword-200',
                   'glove-twitter-100', 'glove-twitter-50']

    models = []
    models_accuracy = []
    for name in model_names:
        models.append(model.Model(name))
    
    print(f"Evaluating Models")
    for model in models:
        accuracy = model.evaluate(synonyms, 4)
        models_accuracy.append(accuracy)

    models_n = ['google300', 'twitter200', 'wiki200',
                'twitter100', 'twitter50', 'humangs', 'randb']

    # Human gold standard from moodle
    models_accuracy.append(0.855)
    # random guessing 1/4
    models_accuracy.append(0.25)

    plotter.Plotter(models_n, models_accuracy)

    print(f"Models: \t{models_n}")
    print(f"accuracies: \t{models_accuracy}")
