# %%
from components import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from fuzzywuzzy import fuzz
from sklearn.base import BaseEstimator

# %%
traindf = pd.read_csv('handcoded.csv')
traindf.head()

# %%
dataset = components.build_initial(traindf['amicus'].values, traindf['bonica'].values)
dataset.shape
dataset.head()

# %%
traindf, testdf = train_test_split(dataset, stratify=dataset['match'])

# %%
model = components.train(traindf)

# %%
predictions = model.predict(testdf[[0, 1]])
(testdf['match'] == predictions).sum()
testdf.shape
model.score(testdf[[0, 1]], testdf['match'])

testdf[testdf['match'] != predictions].shape
confusion_matrix(testdf['match'], predictions)

# %%
dataset.match.sum()

# %%
class FuzzyWuzzy(BaseEstimator):
    """An example of classifier"""

    def __init__(self):

        return None

    def fit(self, X, y=None):

        return self


    def predict_proba(self, X):
        return np.array([fuzz.ratio(x[0], x[1]) for x in X])

    def predict(self, X, fw_param):
        return np.array([self.predict_example(x, fw_param) for x in X])

    def predict_example(self, x, fw_param):
        return 1 if fuzz.ratio(x[0], x[1]) > fw_param else 0



parameters = np.arange(0, 100, 10)


print(parameters)
for p in parameters:

    fw = FuzzyWuzzy()
    fwp = fw.predict(traindf[[0, 1]].values, p)
    print(confusion_matrix(traindf['match'], fwp))


fw = FuzzyWuzzy()
fwp = fw.predict(testdf[[0, 1]].values, 50)
print(confusion_matrix(testdf['match'], fwp))
