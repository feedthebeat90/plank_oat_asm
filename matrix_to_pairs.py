# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv('csvs/handcoded_test.csv', index_col=0)
ind = df.index.values

pairs = np.tril(np.array(np.meshgrid(ind, ind)), -1).T.reshape(-1, 2)
matches = df.to_numpy().flatten()

nudf = pd.DataFrame(data=np.append(pairs, matches[...,None], 1), columns=['0', '1', 'Match'])
nudf = nudf[nudf['0'] != 0]

nudf.to_csv('csvs/handcoded_test_pairs.csv', index=False)
