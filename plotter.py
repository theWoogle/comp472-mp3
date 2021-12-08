import matplotlib.pyplot as plt


class Plotter():
    def __init__(self, x_models, y_accuracies) -> None:
        self.x_models = x_models
        self.y_accuracies = y_accuracies
        self.plot()

    def plot(self):
        plt.bar(self.x_models, self.y_accuracies)
        plt.title("Performance of models")
        plt.ylabel("accuracy $\dfrac{C}{V}$")
        plt.xlabel("models")
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig('results/performance.pdf')
