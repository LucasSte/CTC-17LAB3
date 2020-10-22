
class AprioriClassifier:

    def __init__(self, data, target):
        self.prediction = data.mode()[target][0]