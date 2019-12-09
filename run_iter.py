# %%
from components import *
import sys
import os
import pandas as pd

iter = 0
for item in os.listdir('partitioned'):
    if item.isdigit():
        if int(item) > iter:
            iter = int(item)
iter += 1
print('Running iteration: ', iter)

dataset = pd.read_csv('partitioned/train.csv')
fulldf = build_full(dataset)
model = train(fulldf)
#print(model.cv_results_)
get_predictions(model, 500, iter)
