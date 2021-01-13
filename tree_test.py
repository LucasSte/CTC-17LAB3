import pandas as pd
import os
from ID3 import ID3Algorithm
from apriori import AprioriClassifier
import json

# Read csv file from path
path = os.getcwd()
df = pd.read_csv(path + '/accident_data.csv')

del df['Data']

# Choose variable to classify
tree_builder = ID3Algorithm('Accident Level')

inf_gains = tree_builder.gain_list(df)
print('Information gains:')
for elem in inf_gains:
    print(elem)

print()

# Delete variables with low information gain
del df['Industry Sector']
del df['Employee ou Terceiro']
# del df['Risco Critico']
del df['Countries']
del df['Genre']

tree_builder.build_tree(df)

apriori_class = AprioriClassifier(df, 'Accident Level')

# print(tree_builder.tree)
print('Decision tree:')
print(json.dumps(tree_builder.tree, indent=4))
print()
print('Apriori classifier=> mode of accident levels data: ', apriori_class.prediction)

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

print('Predicted accident level for example: ', tree_builder.predict(input_test))
