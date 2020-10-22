import numpy as np
import pandas as pd

class ID3Algorithm:

    eps = np.finfo(float).eps
    tree = None

    '''

    @:param data: data is the csv variable created by pd.read_csv
    @:param target: the name of the csv field we want to classify. Ex: 'Accident Level'
    '''
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def entropy(self):
        entropy_calc = 0
        values = self.data[self.target].unique()
        for item in values:
            frac = self.data[self.target].value_counts()[item] / len(self.data[self.target])
            entropy_calc -= frac * np.log(frac)

        return entropy_calc

    def entropy_of_attribute(self, attribute):
        target_unique = self.data[self.target].unique()
        variables_unique = self.data[attribute].unique()
        entropy_calc = 0

        for item in variables_unique:
            entropy_aux = 0
            denominator = 0
            for target_item in target_unique:
                numerator = len(self.data[attribute][self.data[attribute] == item][self.data[self.target] == target_item])
                denominator = len(self.data[attribute][self.data[attribute] == item])

                frac = numerator / (denominator + self.eps)
                entropy_aux -= frac*np.log(frac + self.eps)

            frac_aux = denominator / len(self.data)
            entropy_calc -= frac_aux * entropy_aux

        return abs(entropy_calc)

    def find_best(self):
        best = -float("inf")
        data_entropy = self.entropy()
        best_key = None

        for key in self.data.keys():
            if key != self.target:
                entropy_attributes = data_entropy - self.entropy_of_attribute(key)
                if best < entropy_attributes:
                    best = entropy_attributes
                    best_key = key

        return best_key

    def drop_attribute(self, attr, value):
        return self.data[self.data[attr] == value].reset_index(drop=True)

    def _build_tree(self, rtree=None):

        node = self.find_best()

        attrs = np.unique(self.data[node])

        if rtree is None:
            rtree = {}
            rtree[node] = {}

        for item in attrs:
            sub_set = self.drop_attribute(node, item)
            values, count = np.unique(sub_set[self.target], return_counts=True)

            if len(count) == 1:
                rtree[node][item] = values[0]
            else:
                rtree[node][item] = self._build_tree(sub_set)

        return rtree

    def build_tree(self):
        self.tree = self._build_tree()
