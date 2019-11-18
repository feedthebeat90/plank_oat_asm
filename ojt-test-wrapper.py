import numpy as np
import pandas as pd
import textdistance
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def get_data():
    # file1 = input('Please input first filename:\t')
    # file2 = input('Please input second filename:\t')
    file1 = 'csvs/amicus_org_names.csv'
    file2 = 'csvs/bonica_orgs_reduced.csv'
    if not file1.endswith('.csv') and not file2.endswith('.csv'):
        raise Exception('Sorry, please input a CSV file.')
    else:
        first = pd.read_csv(file1)['x'].tolist()
        second = pd.read_csv(file2)['x'].tolist()
    return list(set(first)), list(set(second))

class FuzzyWuzzy(BaseEstimator):
    """An example of classifier"""

    def __init__(self):

        return None

    def fit(self, X, y=None):

        return self


    def predict_proba(self, X):
        return np.array([fuzz.ratio(x[0], x[1]) for x in X])

    def predict(self, X):
        return 1 if np.array([fuzz.ratio(x[0], x[1]) for x in X]) else 0

model = FuzzyWuzzy()
list1, list2 = get_data()

def get_predictions(list1, list2, model, num_matches, samplesize):
    pairs = np.array(list(product(list1, np.random.choice(list2, size = samplesize, replace=False))))
    scores = model.predict_proba(pairs)
    pairs = [(x[0],x[1]) for x in pairs]
    pairs_df = pd.DataFrame({'Pairs': pairs, 'Scores': scores})
    pairs_df = pairs_df.sort_values(by=['Scores'], ascending=False)
    truncated_df = pairs_df.head(num_matches)

    return truncated_df['Pairs']

print(get_predictions(list1, list2, model, 3, 1000))