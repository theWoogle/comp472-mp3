class Model():
    def __init__(self, link: str, model_name:str, voc_size:int) -> None:
        self.model_name = model_name
        self.voc_size = voc_size
        self.link = link
    
    C = 0   # number of correct labels
    V = 0   # number of questions answered w/o guessing

    def download(self) -> None:
        print("\nDownloading")

    def predict(self, dataset) -> None:
        print("\npredict")