# add imports here
import numpy as np
import pandas as pd
import random
import os
import io
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
    def __init__(self, threshold, methods=[textdistance.cosine, 
                                           textdistance.jaccard,
                                           textdistance.sorensen, 
                                           textdistance.tversky, 
                                           textdistance.tanimoto]):
        self.methods = methods
        self.fuzz_threshold = threshold
        print('\n*****************  STRING MATCHER  *****************\n')

    def grab_inputs(self):        
        # input filenames
        f1 = "csvs/" + input("Messy list filename >> ")
        if not f1.endswith(".csv"):
            raise Exception(f"{f1} is not a csv file.")
        f2 = "csvs/" + input("Target list filename >> ")
        if not f2.endswith(".csv"):
            raise Exception(f"{f2} is not a csv file.")            
        files = [f1, f2]

        # try to select string column
        setlists = []
        for f in files:
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

        return setlists[0], setlists[1]

    def preliminary_fuzz(self, messy_list, target_list):
        print(f"\nNum strings in messy list: {len(messy_list)}")
        print(f"Num strings in target list: {len(target_list)}")

        # input number of strings to match
        num_iter = input("Num labeling rounds >> ")
        print("\n* NOTE: WARNING ABOUT MODEL ROBUSTNESS VS. TRAINSET SIZE? *")
        if not int(num_iter):
            raise Exception("Please enter a positive number.")
        else:
            print("\nComputing fuzz ratios...")

            # for each string in sample of shuffled messy list, 
            # for each string in target list, compute fuzz.ratio
            output = []
            count = 1
            random.shuffle(messy_list)
            for i in messy_list[:int(num_iter)]:
                print(f"{count}/{num_iter}", end="\r")
                for j in target_list:
                    score = fuzz.ratio(i,j)
                    # filter out perfect matches and scores below threshold
                    if score >= self.fuzz_threshold and score != 100:
                        output.append([i,j,score])
                    else:
                        next
                count += 1

        print("Initial matching complete.\n")
        return output

    def ask_about_matches(self, first_matches, bins=[0,100]):
        # bin matches and check num high fuzz.ratio pairs
        matches = first_matches.copy()
        matches['fuzzscore'] = pd.to_numeric(matches['fuzzscore'])
        matches['bin'] = pd.cut(x=matches['fuzzscore'], bins=bins)
        bin_counts = matches.groupby(pd.cut(matches['fuzzscore'], bins=bins)).size()
        ### REMOVE PRINTS HERE -- JUST FOR REFERENCE FOR NOW ###
        print()
        print(bin_counts)
        print("\n* NOTE: PERHAPS WE DON'T SHOW THE MATCH COUNT BELOW \n SO AS NOT TO BIAS THE USER, BUT WE SHOULD ALSO MAYBE \n FORCE RESTART preliminary_fuzz WITH A DIFFERENT THRESHOLD \n IF NUM MATCHES ABOVE SCORE TOO HIGH/LOW? *")
        
        ### INSERT SOMETHING ABOUT SAMPLING EVENLY FROM BINS HERE ####

        # check for existing labeled.csv file
        if os.path.isfile("csvs/labeled.csv"):
            stored = pd.read_csv("csvs/labeled.csv")
        else:
            cols=["str1", "str2", "fuzzscore", "match"]
            df = pd.read_csv(io.StringIO(""), names=cols, dtype=dict(zip(cols,[str, str, int, int]))) 
            df.to_csv("csvs/labeled.csv", index=False)
            stored = pd.read_csv("csvs/labeled.csv")

        # for now, use matches df, set as list of tuples and shuffle
        matches = list(matches.itertuples(index=False, name=None))
        random.shuffle(matches)

        for pair in matches:
            # take input from user as match
            str1 = pair[0]
            str2 = pair[1]
            os.system("clear")
            print('Type "exit" to exit matching function.\n(Results will be stored in "labeled.csv")\n\n')
            match = input(f"String 1:\t{str1}\nString 2:\t{str2}\n\n\nDo these match? [y/n] >> ")
            
            # if exit, updated labeled and firstmatches
            if match == "exit":
                print('\nSaving labeled results to "labeled.csv"...')
                stored.to_csv("csvs/labeled.csv", index=False)
                print('Cleaning labeled results from "firstmatches.csv"...')
                first_matches.to_csv("csvs/firstmatches.csv", index=False)
                print("Files updated.")
                return
            # if match, indicate with 1, and transfer row to labeled
            if match == "y":
                pair = list(pair)[:-1]
                pair.append(1)
                stored = stored.append({"str1": pair[0], "str2": pair[1], "fuzzscore": pair[2], "match": pair[3]}, ignore_index=True)
                first_matches = first_matches.drop(first_matches[(first_matches['str1'] == str1) & (first_matches['str2'] == str2)].index)
            # otherwise, indicate with 0, and transfer row to labeled
            elif match == "n":
                pair = list(pair)[:-1]
                pair.append(0)
                stored = stored.append({"str1": pair[0], "str2": pair[1], "fuzzscore": pair[2], "match": pair[3]}, ignore_index=True)
                first_matches = first_matches.drop(first_matches[(first_matches['str1'] == str1) & (first_matches['str2'] == str2)].index)
            # otherwise raise exception
            else:
                raise Exception("Please select y or n.")
        
        stored.to_csv("csvs/labeled.csv", index=False)
        first_matches.to_csv("csvs/firstmatches.csv", index=False)
        print("\nLabeling complete.")

    def train(self, labeled):
        labeled_df = pd.read_csv(labeled)
        print()
        print(labeled_df)

    def run(self):
        # check whether firstmatches.csv (fuzz output) exists yet
        if not os.path.isfile("csvs/firstmatches.csv"):
            self.messy_list, self.target_list = self.grab_inputs()
            first_matches = self.preliminary_fuzz(self.messy_list, self.target_list)
            first_matches = pd.DataFrame(first_matches, columns=["str1", "str2", "fuzzscore"])
            label = input("Begin labeling? [y/n] >> ")
            if label != "n" and label != "y":
                raise Exception("Please select y or n.")
            elif label == 'n':
                print('Saving matches as "firstmatches.csv"...')
                first_matches.to_csv("csvs/firstmatches.csv", index=False)
                print("Matches saved for later.")
                print('\n****************************************************\n')
                return
            else:
                print("Proceeding to hand matching...")
                next
        # messy check to determine whether you want to overwrite existing firstmatches.csv
        else:
            restart = input('"firstmatches.csv" already exists...\nOverwrite and start again? [y/n] >> ')
            if restart != "n" and restart != "y":
                raise Exception("Please select y or n.")
            elif restart == "y":
                sure = input("Are you sure? [y/n] >> ")
                if sure != "n" and sure != "y":
                    raise Exception("Please select y or n.")
                elif sure == "n":
                    print("Resuming hand matching...")
                    next
                else:
                    print('Restarting STRING MATCHER...\nDeleting existing "firstmatch.csv" file...')
                    os.remove("csvs/firstmatches.csv")
                    self.run()
            else:
                print("Resuming hand matching...")
            first_matches = pd.read_csv("csvs/firstmatches.csv")[["str1", "str2", "fuzzscore"]]

        # proceed with firstmatches data to hand labeling
        self.ask_about_matches(first_matches, bins=[self.fuzz_threshold, 90, 95, 99])

        ### INSERT FUNC TO READ IN LABELED FOR TRAINING HERE
        self.train("csvs/labeled.csv")

        print('\n****************************************************\n')

test = StringMatcher(threshold=70)
test.run()