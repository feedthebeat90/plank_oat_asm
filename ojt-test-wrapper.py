import numpy as np
import pandas as pd
import random
import os
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

class StringMatcher:

    def __init__(self, methods=[textdistance.cosine, 
                                textdistance.jaccard,
                                textdistance.sorensen, 
                                textdistance.tversky, 
                                textdistance.tanimoto]):
        self.methods = methods
        print('\n***************  STRING MATCHER  **************\n')

    def grab_inputs(self):
        # input number of files to upload
        num_files = input("Num files [1/2]:  ")

        # if not integer raise exception
        if not int(num_files):
            raise Exception("Please enter 1 or 2.")

        setlists = []

        # if 2 files, input filenames
        if int(num_files) == 2:
            f1 = "csvs/" + input("First filename:  ")
            f2 = "csvs/" + input("Second filename:  ")
            files = [f1, f2]

            # if files are csv, try to select string column
            for f in files:
                if f.endswith(".csv"):
                    f = pd.read_csv(f).select_dtypes(include="object")

                    # if no string column, raise exception
                    if len(f.columns) == 0:
                        raise Exception("No candidate columns identified.")

                    # if more than one string column, raise exception
                    elif len(f.columns) > 1:
                        raise Exception("More than one column identified.")
                
                    # if one string column, collect list set of elements
                    else:
                        for col in f:
                            setlists.append(list(set(f[col].tolist())))

                # if file not csv, raise exception
                else:
                    raise Exception(f"{f} is not a CSV file.")

        # if 1 file, input filename
        elif int(num_files) == 1:
            f = "csvs/" + input("Filename:  ")

            # if file is csv, try to select 2 string columns
            if f.endswith(".csv"):
                f = pd.read_csv(f).select_dtypes(include="object")

                # if no string column, raise exception
                if len(f.columns) == 0:
                    raise Exception("No candidate columns identified.")

                # if one string column, raise exception
                if len(f.columns) == 1:
                    raise Exception("Only one candidate column identified.")

                # if more than two string columns, raise exception
                if len(f.columns) > 2:
                    raise Exception("More than one column identified.")

                # if two string columns, collect list sets of elements
                else:
                    for col in f:
                        setlists.append(list(set(f[col].tolist())))

            # if file not csv, raise exception
            else:
                raise Exception(f"{f} is not a CSV file.")

        return setlists

    def fuzzify(self, setlists):
        # read in set lists and print lengths
        l1, l2 = setlists[0], setlists[1]
        print(f"\nNum strings in list 1:  {len(l1)}")
        print(f"Num strings in list 2:  {len(l2)}")

        # input number of strings to match
        num_iter = input("Num labeling rounds:  ")

        # if not integer raise exception
        if not int(num_iter):
            raise Exception("Please enter a positive number.")
    
        else:
            print("\nComputing fuzz ratios...")

            # for each string in sample of shuffled l2, 
            # for each string in l1, compute fuzz.ratio
            output = []
            count = 1
            random.shuffle(l2)
            for i in l2[:int(num_iter)]:
                print(f"{count}/{num_iter}", end="\r")
                for j in l1:
                    score = fuzz.ratio(i,j)
                    output.append([i,j,score])
                count += 1
        
        ### INSERT BINNING OF FUZZ RATIOS HERE

        # output list of lists of string pairs and fuzz scores
        print("Initial matching complete.\n")

        return output

    def ask_about_matches(self, matches):
        results = []
        random.shuffle(matches)

        # collect strings without fuzz scores and display for labeling
        for pair in matches:
            match = input(f"String 1:  {pair[0]}\nString 2:  {pair[1]}\nDo these match [y/n]?  ")
            
            # clear previous 3 lines
            print ("\033[A                             \033[A")
            print ("\033[A                             \033[A")
            print ("\033[A                             \033[A")
            
            # if match, indicate with 1
            if match == "y":
                pair.append(1)
                results.append(pair)

            # otherwise, indicate with 0
            elif match == "n":
                pair.append(1)
                results.append(pair)

            # otherwise raise exception
            else:
                raise Exception("Please select y or n.")
        
        print("Labeling complete.\n")

        return results

    # def train_test_split(self):


    def run(self):
        self.inputs = self.grab_inputs()
        matches = self.fuzzify(self.inputs)
        labeled = self.ask_about_matches(matches)

        # SHOULD HAVE SOME MECHANISM TO SAVE MATCHES FOR LATER
        # AND PICK UP WHERE WE LEFT OFF
        # label = input("Begin labeling? [Y/N]  ")
        # if label == "N":
        #     print('Saving matches as "firstmatches.csv"...')
        #     df = pd.DataFrame(output, columns=['str1', 'str2', 'fuzzscore'])
        #     df.to_csv('firstmatches.csv')
        #     print("Matches saved for later.")
        # elif label != "Y":
        #     raise Exception("Please enter Y or N.")
        # else:
        #     return output

        print('\n***********************************************\n')

test = StringMatcher()
test.run()



# ask_about_matches(get_predictions(list1, list2, model, sample_size=10))

    # pairs = np.array(list(product(list1, np.random.choice(list2, size = samplesize, replace=False))))
    # scores = model.predict_proba(pairs)
    # pairs = [(x[0],x[1]) for x in pairs]
    # pairs_df = pd.DataFrame({'Pairs': pairs, 'Scores': scores})
    # pairs_df = pairs_df.sort_values(by=['Scores'], ascending=False)
    # truncated_df = pairs_df.head(num_matches)

    # return truncated_df['Pairs']