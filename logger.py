import model


class Logger():
    analysis = open(f"results/analysis.csv", 'a')
    analysis.write(
        f"name, voc-size, # correct, # answered w/o guessing, accuracy")

    def __init__(self, name: str) -> None:
        self.details = open(f"results/{name}-details.csv", 'a')
        self.details.truncate(0)  # clear file
        self.details.write(f'Question, Answer, Guess, Label')

    def append_details_csv(self, question: str, answer: str, guess: str, label: str):
        "append parameters for each question in the Synonym Test dataset, in a single line"
        self.details.write(f'\n{question}, {answer}, {guess}, {label}')

    def append_analysis_csv(self, model: "model.Model"):
        "append model parameters in single line"
        self.analysis.write(
            f"\n{model.name}, {model.voc}, {model.C}, {model.V}, {round(model.C/model.V, 4)}")
