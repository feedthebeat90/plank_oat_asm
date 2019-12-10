# %%
# This file includes the component functions to run the matching algorithm when the initial training set, unmatched string pool, and test set are available.
import os
import numpy as np
import pandas as pd
import textdistance
from itertools import product
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

class stringdist(FunctionTransformer):
    # def __init__(self, methods=[textdistance.cosine, textdistance.jaccard]):
    def __init__(self, methods=[textdistance.cosine, textdistance.jaccard, textdistance.lcsstr, textdistance.tversky, textdistance.levenshtein]):
        self.methods = methods

    def stringdist_wrap(self, row):
        a, b = row[[0, 1]]
        out = pd.Series([m.distance(a, b) for m in self.methods])
        if np.any(np.isinf(out.values)):
            print(a, '---', b)
            print(out)
        return out

    def fit(self, X, y):
        return self

    def transform(self, X):
        #print(X)
        #print(X.apply(self.stringdist_wrap, axis=1))
        res = X.apply(self.stringdist_wrap, axis=1)
        return res

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
    train.to_csv('partitioned/train.csv')
    return train

# %%
# Combines the initial training set with the batches of human labeled
# examples from prior loops (stored at the given path)
def build_full(dataset):
    dfs = [dataset]
    for item in os.listdir('partitioned'):
        if item.isdigit():
            for filename in 'partitioned/' + item:
                if filename.endswith('.csv'):
                    dfs.append(pd.read_csv(path + filename))
                    break
    fulldf = pd.concat(dfs)
    return fulldf

def train(dataset, it):
    pipeline = Pipeline([('stringdist', stringdist()), ('forest', RandomForestClassifier())])

    #the actual model
    parameters = {
        'forest__max_depth': [5, 10, 20],
        'forest__n_estimators': [10, 50, 100]
        }

    GSCV = GridSearchCV(cv = 5,
                       estimator = pipeline,
                       param_grid = parameters,
                       verbose = 2)

    model = GSCV.fit(dataset[['0','1']], dataset['match'])

    os.makedirs('partitioned/' + str(it))

    print("Best parameters:")
    print(model.best_params_)
    print("Feature importances:")
    print(model.best_estimator_.named_steps["forest"].feature_importances_)

    #print(str(model.cv_results_))
    #jsond = json.dumps(model.cv_results_)
    #f2 = open('partitioned/' + str(it) + '/gridsearch.txt', 'w')
    #f2.write(str(model.cv_results_))
    #f2.close()

    return model

def get_predictions(model, num_candidates, num_matches, it):
    bonica = pd.read_csv('csvs/bonica_orgs_reduced.csv')
    amicus = pd.read_csv('csvs/amicus_org_names.csv')
    bonica_sample = bonica[num_candidates*(it-1): num_candidates*(it)]

    pairs = pd.DataFrame(list(product(amicus['amicus'].values, bonica_sample['bonica'].values)), columns=['0', '1'])

    print('Compiled ', pairs.shape[0], ' pairs from unmatched data. Now finding most probable match candidates...')

    # score the pairs and save to a csv for user to validate them
    #pairs = samples['0'].values.reshape(-1,2)
    #pairs = pd.DataFrame(data=pairs, columns=['0', '1'])
    #print(pairs.head())
    preds = model.predict_proba(pairs)
    scores = preds[:,0]
    #print(scores[0:5])
    ind = np.argpartition(scores, -num_matches)[-num_matches:]
    pairs['score'] = scores
    df = pairs.iloc[ind]
    df['match'] = ''
    filename = 'partitioned/' + str(it) + '/labeled.csv'
    os.makedirs('partitioned/' + str(it))
    df.to_csv(filename, index=False)
    print('File ', filename, ' has been created. Validate results using the "match" column and continue with the next iteration.')
