# %%
from components import *
import pandas as pd

# %%
traindf = pd.read_csv('csvs/handcoded.csv')
traindf.head()

# %%
dataset = build_initial(traindf['amicus'].values, traindf['bonica'].values)
print(dataset.shape)
dataset.to_csv('partitioned/train.csv', index=False)
