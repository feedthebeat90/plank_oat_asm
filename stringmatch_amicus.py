# %%
import numpy as np
import pandas as pd
import textdistance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# %%
# Read in preprocessed data (skips first block of code in R file)
handcoded = pd.read_csv('handcoded.csv')
handcoded = handcoded.drop(['Unnamed: 0'], axis=1)

# %%
# Vector of amicus strings concatenated with corresponding bonica strings
handcoded_vec = handcoded['amicus'].map(str) + '_' + handcoded['bonica']

# %%
# Create a set of incorrect matches
# First, copy the correct matches
tmp = handcoded.copy()
# Shuffle the amicus column - makes most of them mismatched
tmp['amicus'] = np.random.permutation(tmp['amicus'].values)
# For any that might still be correct matches, filter them out
# by making sure the concatenated string isn't in the vector of
# correct concatenated strings (handcoded_vec)
tmp_vec = tmp['amicus'].map(str) + '_' + tmp['bonica']
tmp = tmp[~tmp_vec.map(lambda x: handcoded_vec.str.contains(x).any())]
tmp['match'] = 0

# %%
# Get one more batch of incorrect matches
tmp2 = handcoded.copy()
tmp2['amicus'] = np.random.permutation(tmp2['amicus'].values)
tmp2_vec = tmp2['amicus'].map(str) + '_' + tmp2['bonica']
tmp2 = tmp2[~tmp2_vec.map(lambda x: handcoded_vec.str.contains(x).any())]
tmp2['match'] = 0

# %%
print(tmp.shape)
print(tmp2.shape)
print(handcoded.shape)

# %%
# Concatenate the incorrect ones, drop duplicates, and concatenate with the correct ones
tmp_full = pd.concat([tmp, tmp2])
tmp_full.drop_duplicates(inplace=True)
train = pd.concat([handcoded, tmp_full])
train['amicus'] = train['amicus'].str.lower()
train['bonica'] = train['bonica'].str.lower()

# %%
# Add more distance metrics?
methods = [textdistance.cosine, textdistance.jaccard]
def stringdist_wrap(row):
    a, b = row[['amicus', 'bonica']]
    out = pd.Series([m.distance(a, b) for m in methods])
    return out

# %%
df = train.apply(stringdist_wrap, axis=1)
df.columns = ['cosine', 'jaccard']

# %%
#the actual model
parameters = {'max_depth': [2,3,4]}

GSCV = GridSearchCV(cv = 5,
                   estimator = RandomForestClassifier(),
                   param_grid = parameters)

model = GSCV.fit(df, train['match'])

preds = model.predict(df)
labels = train['match']

accuracy_score(preds, labels)
