

class Logger():
    def __init__(self, name) -> None:
        self.details = open(f"{name}-details.csv",'a')
        self.details.write(f'Question, Answer, Guess, Label')

    def append_details_csv(self, question: str, answer: str, guess: str, label: str):
        "append paramters for each question in the Synonym Test dataset, in a single line"
        self.details.write(f'\n{question}, {answer}, {guess}, {label}')

    # def append_analysis_csv(self, params: Model):
    #     "append model parameters in single line"
