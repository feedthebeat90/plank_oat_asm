import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process

def fuzzywuzzy(word1, word2):
    return fuzz.ratio(word1, word2) 

def get_predictions(list1, list2, model, num_matches):
    pairs = []
    scores = []
    word_count = 0
    for word1 in list1:
        word_count += 1
        for word2 in list2:
            pairs.append((word1,word2))
            scores.append(model(word1, word2))
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