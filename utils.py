import matplotlib.pyplot as plt
import numpy as np

class Downloader():
    def __init__(self) -> None:
        pass

class Plotter():
    def __init__(self, x_models,y_accuracies) -> None:
        self.x_models= x_models
        self.y_accuracies = y_accuracies
        self.plot()
        # pass

    def plot(self):
        plt.bar(self.x_models, self.y_accuracies)
        # plt.show()
        plt.savefig('performance.pdf')

