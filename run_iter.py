# %%
from Components import *
import sys
import os
import pandas as pd
from datetime import datetime
from sklearn import metrics

it = 0
for item in os.listdir('partitioned'):
    if item.isdigit():
        if int(item) > it:
            it = int(item)
it += 1
print(datetime.now().strftime("%H:%M:%S"))
print('Running iteration: ', it)

dataset = pd.read_csv('partitioned/train.csv')
fulldf = build_full(dataset)
model = train(fulldf, it)
#print(model.cv_results_)
print(datetime.now().strftime("%H:%M:%S"))
get_predictions(model, 50, 100, it)

# Two test sets

# representative
print("Representative")
test1 = pd.read_csv('csvs/outputs/testpairs.csv')
true1 = test1['Match']
preds1 = model.predict(test1[['Str1', 'Str2']])
test1['prediction'] = preds1
test1.to_csv('results/representative.csv')
print("Precision: ", metrics.precision_score(true1, preds1))
print("Recall: ", metrics.recall_score(true1, preds1))
print()

# match dense
print("Match Dense")
test2 = pd.read_csv('csvs/handcoded_test_pairs.csv')
true2 = test2['Match']
preds2 = model.predict(test2[['0', '1']])
test2['prediction'] = preds2
test2.to_csv('results/match_dense.csv')
print("Precision: ", metrics.precision_score(true2, preds2))
print("Recall: ", metrics.recall_score(true2, preds2))
print()
