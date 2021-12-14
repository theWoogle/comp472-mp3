import gensim.downloader as api
import gensim.utils
import os.path
import numpy as np
import gensim.similarities
import random
import logger

random.seed(42)


class Model():
    def __init__(self, name: str) -> None:
        self.name = name
        self.C = 0          # number of correct labels
        self.V = 0          # number of questions answered w/o guessing
        self.download()
        self.l = logger.Logger(name)

    def download(self) -> None:
        print(f"Downloading {self.name}")
        if not os.path.exists(f'data/{self.name}.d2v'):
            self.model = api.load(self.name)
            self.model.save(f'data/{self.name}.d2v')
        else:
            self.model = gensim.utils.SaveLoad.load(f'data/{self.name}.d2v')
        self.voc = len(self.model)
        return

    def evaluate(self, dataset: np.ndarray, feature_len: int) -> None:
        """
        evaluate cosine-similarity of words in first column of `dataset` with words in columns 2 - 2 + `feature_len`   
        guesses randomly if word if question is not in vocabulary.  
        increments C and V
        """
        # vector to store cosine values for each possible synonym
        cosines = np.empty((len(dataset), feature_len))
        for i in range(len(dataset)):
            for j in range(feature_len):
                # handles key error when word is not in vocabulary
                try:
                    cosines[i, j] = round(self.model.similarity(
                        dataset[i, 0], dataset[i, 2+j]), 4)
                except Exception as e:
                    # guess probabilty later if word is unseen
                    # will result in 4x -1 if guess word raises exception
                    cosines[i, j] = -1
                    pass                

            # guess probability if all words or question-word are unknown = -1
            if all([round(x, 4) < 0 for x in cosines[i, :]]):
                idx_guess = random.randint(0, 3)
                label = 'guess'
            else:
                # no guess -> increase V
                self.V += 1
                # select index with highest cosine/similarity
                idx_guess = np.argmax(cosines[i, :])
                # check if answer matches  provided solution
                if dataset[i, idx_guess+2] == dataset[i, 1]:
                    self.C += 1
                    label = 'correct'
                else:
                    label = 'wrong'
            self.l.append_details_csv(
                dataset[i, 0], dataset[i, 1], dataset[i, idx_guess+2], label)

        self.l.append_analysis_csv(self)
        return round(self.C/self.V, 4)
