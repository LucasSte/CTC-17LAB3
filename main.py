import pandas as pd
import os
from model import Model

path = os.getcwd()
df = pd.read_csv(path + '/accident_data.csv')

model = Model(df)
model.drop_unrealistic_assumptions()
model.drop_lightweight_columns()
model.slice_data_frame()
model.id3()
# model.show_tree()
model.apriori()