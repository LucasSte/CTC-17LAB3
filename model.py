from numpy.lib.shape_base import column_stack
from numpy.random import RandomState
from apriori import AprioriClassifier
from ID3 import ID3Algorithm
import json
from sklearn.metrics import mean_squared_error, cohen_kappa_score, confusion_matrix
from math import sqrt

class Model:

    def __init__(self, df):
        self.df = df
        self.train = None
        self.test = None
        self.tree_builder = ID3Algorithm('Accident Level')
        self.actual = []
        self.predicted_id3 = []
        self.predicted_apriori = []
        self.actual_values = None

    def slice_data_frame(self):
        n = len(self.df)
        rng = RandomState()
        self.train = self.df.sample(frac=0.8, random_state=rng)
        self.test = self.df.loc[~self.df.index.isin(self.train.index)]
        print("Data frame has " + str(n) + " rows in total")
        print("{:>30}".format(str(len(self.train)) + " rows for training (80%)"))
        print("{:>30}".format(str(len(self.test)) + " rows for training (20%)"))
        self._set_test_values()
        return self.train, self.test

    def drop_unrealistic_assumptions(self):
        del self.df['Data']

    def drop_lightweight_columns(self):
        columns = self.tree_builder.gain_list(self.df)
        for i in range(0,4):
            del self.df[columns[i][0]]

    def show_tree(self):
        print(json.dumps((self.tree_builder).tree, indent=4))

    def _get_val(self, val):
        val = val.strip()
        if (val == 'I'): 
            return 1
        if (val == 'II'): 
            return 2
        if (val == 'III'): 
            return 3
        if (val == 'IV'): 
            return 4
        if (val == 'V'): 
            return 5
        if (val == 'VI'): 
            return 6
        print(val)

    def _set_test_values(self):
        for row in self.test.values:
            input_test = {
                'Local': row[0],
                'Accident Level': row[1],
                'Potential Accident Level': row[2],
                'Risco Critico': row[3]
            }

            self.actual.append(input_test)
            self.actual_values = [self._get_val(x['Accident Level']) for x in self.actual]

    def id3(self):
        self.tree_builder.build_tree(self.train)
        total = len(self.test)
        correct = 0
        incorrect = 0
        for input_test in self.actual:
            prediction = self.tree_builder.predict(input_test)

            self.predicted_id3.append(self._get_val(prediction))

            if (prediction is input_test['Accident Level']):
                correct += 1
            else: incorrect += 1

        print("\n-----------------ID3 Algorithm Report-----------------")
        print("\n       #unknown = {}".format(self.tree_builder.unknown))
        print("       #correct/total = {}/{} = {:.2f}%".format(correct, total, correct/total * 100))
        print("       #incorrect/total = {}/{} = {:.2f}%".format(incorrect, total, incorrect/total * 100))
        print("       root mean square error: {:.2f}".format(sqrt(mean_squared_error(self.actual_values, self.predicted_id3))))
        print("       kappa score: {:.6f}\n".format(cohen_kappa_score(self.actual_values, self.predicted_id3, labels=None, weights=None)))
        print("\nconfusion matrix: \n{}".format(confusion_matrix(self.actual_values, self.predicted_id3)))
        
    def apriori(self):
        total = len(self.test)
        correct = 0
        incorrect = 0
        prediction = AprioriClassifier(self.train, 'Accident Level').prediction
        for input_test in self.actual:
            self.predicted_apriori.append(self._get_val(prediction))
        
            if (prediction is input_test["Accident Level"]):
                correct += 1
            else: incorrect += 1

        print("\n--------------Apriori Classifier Report---------------")
        print("\n       #correct/total = {}/{} = {:.2f}%".format(correct, total, correct/total * 100))
        print("       #incorrect/total = {}/{} = {:.2f}%".format(incorrect, total, incorrect/total * 100))
        print("       root mean square error: {:.2f}".format(sqrt(mean_squared_error(self.actual_values, self.predicted_apriori))))
        print("       kappa score: {:.6f}\n".format(cohen_kappa_score(self.actual_values, self.predicted_apriori, labels=None, weights=None)))
        print("\nconfusion matrix: \n{}".format(confusion_matrix(self.actual_values, self.predicted_apriori)))
    
