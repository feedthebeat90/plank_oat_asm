# %%
import numpy as np
import pandas as pd
import textdistance
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

class stringdist(FunctionTransformer):
    def __init__(self, methods=[textdistance.cosine, textdistance.jaccard]):
        self.methods = methods

    def stringdist_wrap(self, row):
        a, b = row[[0, 1]]
        out = pd.Series([m.distance(a, b) for m in self.methods])
        return out

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.apply(self.stringdist_wrap, axis=1)

# %%
def build_initial(x, y):
    matches = pd.concat([x, y], axis=1)
    matches['match'] = 1
    match_vec = x + '_' + y

    # Create a set of incorrect matches
    # First, copy the correct matches
    tmp = matches.copy()
    print(tmp.head())
    # Shuffle the first column - makes most of them mismatched
    tmp[0] = np.random.permutation(tmp[0].values)
    # For any that might still be correct matches, filter them out
    # by making sure the concatenated string isn't in the vector of
    # correct concatenated strings (match_vec)
    tmp_vec = tmp[0].map(str) + '_' + tmp[1]
    tmp = tmp[~tmp_vec.map(lambda s: match_vec.str.contains(s).any())]
    tmp['match'] = 0

    # Get one more batch of incorrect matches
    tmp2 = matches.copy()
    tmp2[0] = np.random.permutation(tmp2[0].values)
    tmp2_vec = tmp2[0].map(str) + '_' + tmp2[1]
    tmp2 = tmp2[~tmp2_vec.map(lambda s: match_vec.str.contains(s).any())]
    tmp2['match'] = 0

    # Concatenate the incorrect ones, drop duplicates, and concatenate with the correct ones
    tmp_full = pd.concat([tmp, tmp2])
    tmp_full.drop_duplicates(inplace=True)
    train = pd.concat([matches, tmp_full])
    train[0] = train[0].str.lower()
    train[1] = train[1].str.lower()
    return train

def train(dataset):
    pipeline = Pipeline([('stringdist', stringdist()), ('forest', RandomForestClassifier())])

    #the actual model
    parameters = {'forest__max_depth': [2,3,4]}

    GSCV = GridSearchCV(cv = 5,
                       estimator = pipeline,
                       param_grid = parameters)

    model = GSCV.fit(dataset[[0, 1]], dataset['match'])
    return model

def get_predictions(x, y, model, num_matches):
    pairs = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
    scores = model.predict_proba(pairs)
    ind = np.argpartition(scores, -num_matches)[-num_matches:]
    return (pairs[ind], scores[ind])

#the interface with the user who asks them whether the predicted matches
#are actually a match. Returns a dictionary with keys that are the pairs
#and values that are 1 or 0 if there is or is not a match
def ask_about_matches(match_pairs):
    results = []
    for pair in match_pairs:
        match = ''
        while match != 'y' and match != 'n':
            print(pair)
            match = input("Do these match (y or n): ")

        if match == 'y':
            results.append(1)
        elif match == 'n':
            results.append(0)
    return results

# %%
amicus = pd.read_csv('amicus_org_names.csv')['x']
bonica = pd.read_csv('bonica_orgs_reduced.csv')['x']

# %%
train = pd.read_csv('data_viable_train.csv')
train.head()

# %%
model = train_initial(train['amicus'].values, train['bonica'].values)
