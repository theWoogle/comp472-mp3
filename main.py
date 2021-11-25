import model
import numpy as np
from logger import Logger

if __name__ == "__main__":

    # Download models --> Takes some minutes
    google_300 = model.Model('word2vec-google-news-300')    

    # Download datasets
    synonyms = np.loadtxt('data/synonyms.csv', dtype=str, delimiter=',', skiprows=1)

    # Evaluate model 1 word2vec-google-news-300
    google_300.evaluate(synonyms, 4)

    # Evaluate model 2 e.g. twitter-300
    # Evaluate model 3 e.g. gigaword-300
    # Evaluate model 4 e.g. wikipedia-300
    # Evaluate model 5 e.g. wikipedia-1024
    # Plot results