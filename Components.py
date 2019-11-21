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
    # def __init__(self, methods=[textdistance.cosine, textdistance.jaccard]):
    def __init__(self, methods=[textdistance.cosine, textdistance.jaccard, textdistance.sorensen, textdistance.tversky, textdistance.tanimoto]):
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
    matches = pd.DataFrame({0: x, 1: y})
    matches['match'] = 1
    match_vec = pd.Series(x + '_' + y)

    # Create a set of incorrect matches
    # First, copy the correct matches
    tmp = matches.copy()
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

def get_predictions(model, num_matches, iter):
    # load unmatched strings and sample up to 20000
    # (to make 10000 pairs)
    strings = np.loadtxt('csvs/trainpool.csv', delimiter=',', skiprows=1)
    n_samples = max(strings.shape[0] - strings.shape[0]%2, 20000)
    samples = np.random.choice(strings, n_samples, replace=True)

    # remove the samples from the unmatched string pool
    # and resave trainpool.csv
    strings = np.setdiff1d(strings, np.unique(samples))
    # RESAVE strings TO csvs/trainpool.csv

    # score the pairs and save to a csv for user to validate them
    pairs = samples.reshape(-1,2)
    scores = model.predict_proba(pairs)
    ind = np.argpartition(scores, -num_matches)[-num_matches:]
    df = pd.DataFrame(data=np.hstack(pairs[ind], scores[ind].T), columns=['0', '1', 'score'])
    df['match'] = ''
    filename = 'csvs/labeled_' + str(iter) + '.csv'
    df.to_csv(filename)
    print('File ', filename, ' has been created. Validate results using the "match" column and continue with the next iteration.')
