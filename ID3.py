import numpy as np
import pandas as pd

class ID3Algorithm:

    eps = np.finfo(float).eps
    tree = None

    '''

    @:param data: data is the csv variable created by pd.read_csv
    @:param target: the name of the csv field we want to classify. Ex: 'Accident Level'
    '''
    def __init__(self, target):
        self.target = target

    def entropy(self, data):
        entropy_calc = 0
        values = data[self.target].unique()
        for item in values:
            frac = data[self.target].value_counts()[item] / len(data[self.target])
            entropy_calc -= frac * np.log(frac)

        return entropy_calc

    def entropy_of_attribute(self, attribute, data):
        target_unique = data[self.target].unique()
        variables_unique = data[attribute].unique()
        entropy_calc = 0

        for item in variables_unique:
            entropy_aux = 0
            denominator = 0
            for target_item in target_unique:
                numerator = len(data[attribute][data[attribute] == item][data[self.target] == target_item])
                denominator = len(data[attribute][data[attribute] == item])

                frac = numerator / (denominator + self.eps)
                entropy_aux -= frac*np.log(frac + self.eps)

            frac_aux = denominator / len(data)
            entropy_calc -= frac_aux * entropy_aux

        return abs(entropy_calc)

    def find_best(self, data):
        best = -float("inf")
        data_entropy = self.entropy(data)
        best_key = None

        for key in data.keys():
            if key != self.target:
                entropy_attributes = data_entropy - self.entropy_of_attribute(key, data)
                if best < entropy_attributes:
                    best = entropy_attributes
                    best_key = key

        return best_key

    def drop_attribute(self, attr, value, data):
        return data[data[attr] == value].reset_index(drop=True).drop(columns=[attr])

    def _build_tree(self, data):

        node = self.find_best(data)
        if node is None:
            return data[self.target].mode()[0]

        attrs = np.unique(data[node])
        rtree = {node: {}}

        for item in attrs:
            sub_set = self.drop_attribute(node, item, data)
            values, count = np.unique(sub_set[self.target], return_counts=True)

            if len(count) == 1:
                rtree[node][item] = values[0]
            else:
                rtree[node][item] = self._build_tree(sub_set)

        return rtree

    def build_tree(self, data):
        self.tree = self._build_tree(data)
