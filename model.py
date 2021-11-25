import gensim.downloader as api
import gensim.utils
import os.path
class Model():
    def __init__(self, name:str) -> None:
        self.name = name
        C = 0   # number of correct labels
        V = 0   # number of questions answered w/o guessing
        self.download()
     
    def download(self) -> None:
        print("\nDownloading")
        if not os.path.exists(f'data/{self.name}.d2v'):
            self.model = api.load(self.name)
            self.model.save(f'data/{self.name}.d2v')
        else:
            self.model = gensim.utils.SaveLoad.load(f'data/{self.name}.d2v')
        self.voc = len(self.model)
        return

    def predict(self, dataset) -> None:
        print("\npredict")