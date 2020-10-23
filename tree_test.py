import pandas as pd
import os
from ID3 import ID3Algorithm
from apriori import AprioriClassifier
import json

path = os.getcwd()
df = pd.read_csv(path + '/accident_data.csv')

del df['Data']

tree_builder = ID3Algorithm('Accident Level')

print(tree_builder.gain_list(df))

del df['Industry Sector']
del df['Employee ou Terceiro']
# del df['Risco Critico']
del df['Countries']
del df['Genre']

tree_builder.build_tree(df)

apriori_class = AprioriClassifier(df, 'Accident Level')

# print(tree_builder.tree)
print(json.dumps(tree_builder.tree, indent=4))
print(apriori_class.prediction)

input_test = {
    'Potential Accident Level': 'III',
    'Accident Level': 'II',
    'Local': 'Local_04',
    'Genre': 'Male',
    'Data': '2016-01-01 00:00:00',
    'Countries': 'Country_01',
    'Industry Sector': 'Metals',
    'Employee ou Terceiro': 'Employee',
    'Risco Critico': 'Cut'
}

print(tree_builder.predict(input_test))
