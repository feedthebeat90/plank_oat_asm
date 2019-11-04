import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.base import BaseEstimator
from itertools import product

#class which makes FuzzyWuzzy act like an aklearn model
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

#function that takes as input two lists of strings and outputs the most likely
#matches using scores from the passed-in model
def get_predictions(list1, list2, model, num_matches):
    pairs = np.array(list(product(list1,list2)))
    scores = model.predict_proba(pairs)
    pairs = [(x[0],x[1]) for x in pairs]
    pairs_df = pd.DataFrame({'Pairs': pairs, 'Scores': scores})
    pairs_df = pairs_df.sort_values(by=['Scores'], ascending=False)
    truncated_df = pairs_df.head(num_matches)

    return truncated_df['Pairs']

#the interface with the user who asks them whether the predicted matches
#are actually a match. Returns a dictionary with keys that are the pairs
#and values that are 1 or 0 if there is or is not a match
def ask_about_matches(match_pairs):
    match_dct = {}
    for pair in match_pairs:
        match = ''
        while match != 'y' and match != 'n':
            print(pair)
            match = input("Do these match (y or n): ")

        if match == 'y':
            match_dct[pair] = 1
        elif match == 'n':
            match_dct[pair] = 0

    return match_dct
