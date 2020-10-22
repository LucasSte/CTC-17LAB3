import pandas as pd
import os
from ID3 import ID3Algorithm
from apriori import AprioriClassifier

path = os.getcwd()
csv = pd.read_csv(path + '/accident_data.csv')

del csv['Data']
del csv['Industry Sector']
del csv['Employee ou Terceiro']
del csv['Risco Critico']
del csv['Countries']

tree_builder = ID3Algorithm('Accident Level')

tree_builder.build_tree(csv)
apriori_class = AprioriClassifier(csv, 'Accident Level')

print(tree_builder.tree)
print(apriori_class.prediction)
