# Imports
import numpy as np
import pandas as pd
import textdistance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import fuzzywuzzy
from fuzzywuzzy import fuzz
from collections import defaultdict

# Read full amicus, 'reduced' bonica and print initial stats
amicus = pd.read_csv('amicus_org_names.csv').drop(['Unnamed: 0'], axis=1).rename(columns={'x': 'amicus'})
bonica = pd.read_csv('bonica_orgs_reduced.csv', header=None, names=['index', 'bonica']).drop(['index'], axis=1)
amicus['amicus'] = amicus['amicus'].apply(lambda x: x.lower())
bonica['bonica'] = bonica['bonica'].apply(lambda x: x.lower())
# print('Starting length of Amicus dataset: {} rows'.format(len(amicus)))
# print('Starting length of Bonica dataset: {} rows'.format(len(bonica)))
# print('Amicus dataset has {} unique elements'.format(len(sorted(list(set(amicus['amicus']))))))
# print('Bonica dataset has {} unique elements'.format(len(sorted(list(set(bonica['bonica']))))))
# print('There are {} exact matches between the Amicus and Bonica datasets'.format(len(set(amicus['amicus']).intersection(bonica['bonica']))))
total_set = set(amicus['amicus']).union(set(bonica['bonica']))
# print('The union set contains {} elements'.format(len(sorted(list(total_set)))))

# Read in handcoded subset (matches between amicus, bonica) and print initial stats
handcoded = pd.read_csv('handcoded.csv')
handcoded = handcoded.drop(['Unnamed: 0'], axis=1)
handcoded['amicus'] = handcoded['amicus'].apply(lambda x: x.lower())
handcoded['bonica'] = handcoded['bonica'].apply(lambda x: x.lower())
handcoded_subset = set(handcoded['amicus']).union(set(handcoded['bonica']))
# print('Starting length of handcoded Amicus-Bonica dataset: {} rows'.format(len(handcoded)))
# print('Handcoded dataset has {} unique elements'.format(len(handcoded_subset)))

# Get set of elements not contained in handcoded subset
unmatched_set = total_set - handcoded_subset
# print('Set of Bonica elements that have not been matched to Amicus elements contains {} rows'.format(len(unmatched_set)))

# Update amicus and bonica by removing handcoded strings
amicus_updated = amicus[~amicus['amicus'].isin(sorted(list(handcoded_subset)))]
bonica_updated = bonica[~bonica['bonica'].isin(sorted(list(handcoded_subset)))]

# print('Length of Amicus dataset post-removal: {}'.format(len(amicus_updated)))
# print('Length of Bonica dataset post-removal: {}'.format(len(bonica_updated)))

# Shuffle and reset index, then combine
amicus_updated_shuffled = amicus_updated.sample(frac=1).reset_index(drop=True)
bonica_updated_shuffled = bonica_updated.sample(frac=1).reset_index(drop=True)

amicus_strings = list(amicus_updated_shuffled['amicus'])
bonica_strings = list(bonica_updated_shuffled['bonica'])

combiner = []
for i in range(len(amicus_updated_shuffled)):
    combiner.append([amicus_updated_shuffled.iloc[i].values[0], bonica_updated_shuffled.iloc[i].values[0]])
combined = pd.DataFrame(combiner, columns=['amicus', 'bonica'])
combined['match'] = np.nan

# bonica_strings = list(combined['bonica'].unique())
# amicus_strings = list(combined['amicus'].unique())

# print('length amicus: '+str(len(amicus_strings)))
# print('length bonica: '+str(len(bonica_strings)))

to_verify = []

num = 1

for i in bonica_strings[:30000]:
    print(str(num)+'\t'+str(i))
    min_score = 85
    max_score = -1
    max_name = ''
    for j in amicus_strings:
        fuzzscore = fuzz.ratio(i,j)
        if (fuzzscore > max_score) and (fuzzscore > min_score) and fuzzscore != 100:
            max_score = fuzzscore
            max_name = j
            to_verify.append([str(i), str(max_name), str(max_score)])
    num += 1

to_verify_df = pd.DataFrame(to_verify, columns=['bonica', 'amicus', 'fuzzscore'])
to_verify_df['match'] = np.nan
to_verify_df.to_csv('to_verify.csv')

# Check for stray handcoded strings, and confirm that strings come from correct sources
# strays = []
# for i in sorted(list(handcoded_subset)):
#     if (i in combined['amicus']) or (i in combined['bonica']): 
#         strays.append(i)
# if len(strays) != 0: print('Stray matched string found!')
# else: print('Handcoded strings successfully removed!')

# for i in combined['bonica']:
#     if (i in amicus['amicus']) or (i in amicus_updated['amicus']) or (i in amicus_updated_shuffled['amicus']):
#         print('Error: string originally from bonica found in amicus column')
# for i in combined['amicus']:
#     if (i in bonica['bonica']) or (i in bonica_updated['bonica']) or (i in bonica_updated_shuffled['bonica']):
#         print('Error: string originally from amicus found in bonica column')
    
# Write CSV of new viable validation and test data
# combined.to_csv('data_viable_test.csv')

# Rename/rewrite handcoded match data to CSV
# handcoded.to_csv('data_viable_train.csv')

# Append handcoded and new cleaned data to create CSV of full dataset
# full_dataset = combined.append(handcoded)
# full_dataset.to_csv('data_all.csv')
# print('Number of rows in viable validation/test data: {}'.format(len(combined)))
# print('Number of rows in hand-matched training data: {}'.format(len(handcoded)))
# print('Number of rows in the full dataset: {}'.format(len(full_dataset)))