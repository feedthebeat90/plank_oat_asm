import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process
from sklearn.base import BaseEstimator
from itertools import product

class FuzzyWuzzy(BaseEstimator):  
    """An example of classifier"""

    def __init__(self):
        
        return None

    def fit(self, X, y=None):

        return self
    
    #assumes x is a tuple
    def fuzwuz(x):
        return fuzz.ratio(x[0],x[1])
        

    def predict_proba(self, X):
        return np.array([fuzz.ratio(x[0], x[1]) for x in X])

def get_predictions(list1, list2, model, num_matches):
    pairs = np.array(list(product(list1,list2)))
    scores = model.predict_proba(pairs)
    pairs = [(x[0],x[1]) for x in pairs]
    pairs_df = pd.DataFrame({'Pairs': pairs, 'Scores': scores})
    pairs_df = pairs_df.sort_values(by=['Scores'], ascending=False)
    truncated_df = pairs_df.head(num_matches)
    
    return truncated_df['Pairs']

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