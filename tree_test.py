import pandas as pd
import os
from ID3 import ID3Algorithm

path = os.getcwd()
csv = pd.read_csv(path + '/accident_data.csv')

del csv['Data']
del csv['Industry Sector']
del csv['Employee ou Terceiro']
del csv['Risco Critico']

tree_builder = ID3Algorithm(csv, 'Accident Level')

tree_builder.build_tree()

print(tree_builder.tree)