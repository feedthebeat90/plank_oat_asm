# --- Add imports here --- #
import numpy as np
import pandas as pd
import random
import os
import io

import sys
sys.path.append("./")
from tfidfmethod import StringMatch

class MatchProgram():

    def __init__(self):
        """
        Grabs data directory and creates subdirectory for csv outputs.
        """
        print("\n***** MatchProgram *****\n")
        self.datadir = input("Directory >> ")
        if not self.datadir.endswith("/"):
            self.datadir += "/"
        if not os.path.exists(self.datadir):
            print("Could not locate directory.")
            self.__init__()
        if not os.path.exists(self.datadir + "outputs/"):
            os.makedirs(self.datadir + "outputs/")

    def strip_file(self, fn):
        """
        Strips csv with single col of strs into set of strs.
        (Called on each input file in load_files.)
        """
        df = pd.read_csv(fn).select_dtypes(include="object")
        if len(df.columns) != 1:
            print("Make sure each file contains 1 column.")
        return [list(set(df[col].tolist())) for col in df][0]

    def load_files(self):
        """
        Creates set of all strs from all specified input files.
        """
        allstrs = []
        fns = [self.datadir + fn for fn in input("Filenames >> ").split()]
        for fn in fns:
            if not os.path.isfile(fn):
                print("Could not locate one or more files.")
                self.load_files()
            if not fn.endswith(".csv"):
                print("Please input CSV files only.")
                self.load_files()
            allstrs.extend(self.strip_file(fn))
        self.unprocessed = list(set(allstrs))
        print(f"\nNum input files: {len(fns)}")
        print(f"Len unprocessed corpus: {len(self.unprocessed)}")
        df = pd.DataFrame({"UnprocessedStrings": self.unprocessed})
        df.to_csv(self.datadir + "outputs/unprocessed.csv", index=False)
        print("Stored as: unprocessed.csv")
        return self.unprocessed

    def preprocess(self, str_lst):
    """
    Takes in unprocessed str set and outputs processed str set.
    """
        str_set = list(set(str_lst)
        str_set = [x.lower() for x in str_set]

        to_be_removed = [",",'"', "_", "-"]
        for char in to_be_removed:
            str_set = [x.replace(char,'') for x in str_set]

        to_be_saved = pd.DataFrame(str_set)
        to_be_saved.to_csv('processed.csv')

        print("\nLen processed corpus: ")
        print("Stored as: processed.csv")
        return str_set

    def grab_alph_chunk(self, l, n):
        """
        Alphabetizes lst, grabs random idx and creates contiguous chunk of size n.
        (Called in provide_test_matrix.)
        """
        start_idx = random.randint(0, len(l)-n-1)
        return sorted(l)[start_idx:start_idx+n]

    # once preprocess is written,
    # replace refs to self.unprocessed with self.processed
    def provide_test_matrix(self):
        """
        Extracts set of test strs from processed str set (removes from full set),
        Outputs empty csv matrix of size (num test strs x num test strs).
        """
        # for now: hardcode self.testsize=36,
        #          figure out reasonable way to calculate
        #          as func of len(self.processed)
        self.testsize = 36
        print(f"\nAmt corpus to test: {round((self.testsize)**2/len(self.unprocessed)*100, 2)}%")
        # for now: sort strs alphabetically,
        #          pick random starting idx
        #          take chunk starting at this idx
        #          (to try to get some matches)
        self.teststrs = self.grab_alph_chunk(self.unprocessed, self.testsize)
        tmp_processed = pd.read_csv(self.datadir + "outputs/unprocessed.csv")
        tmp_processed = tmp_processed[~tmp_processed["UnprocessedStrings"].isin(self.teststrs)]
        tmp_processed.to_csv(self.datadir + "outputs/trainpool.csv", index=False, header=False)
        testmatrix = np.empty((self.testsize, self.testsize))
        testmatrix[:] = np.nan
        testdf = pd.DataFrame(testmatrix, index=self.teststrs, columns=self.teststrs)
        testdf.to_csv(self.datadir + "outputs/testmatrix_unlabeled.csv")
        print("Empty test set generated")
        print("Stored as: testmatrix_unlabeled.csv")
        print("\nPlease label matches as 1, nonmatches as 0")
        print("When you have labeled the full test set, rename file testmatrix_labeled.csv")

    def read_test_matrix(self):
        """
        Takes in labeled test str matrix and flattens into pairwise csv.
        """
        test = pd.read_csv(self.datadir + "outputs/testmatrix_labeled.csv", index_col=0)
        if not list(np.unique(test)) == [0, 1]:
            print("Labels must be 0 or 1")
            self.read_test_matrix()
        flat_test = []
        for col in test:
            for row in test[col].iteritems():
                flat_test.append([row[0], col, row[1]])
        flat_test_df = pd.DataFrame(flat_test, columns=["Str1", "Str2", "Match"])
        flat_test_df.to_csv(self.datadir + "outputs/testpairs.csv", index=False)
        print("Test matrix flattened")
        print("Stored as: testpairs.csv")
        print("\nTest class distribution:")
        print(flat_test_df["Match"].value_counts())

    def train_match_interface(self):
        """
        Samples pairs from prelim tf-idf score bins and feeds to user via command line.
        """
        self.trainpool = pd.read_csv(self.datadir + "outputs/trainpool.csv")["UnprocessedStrings"].tolist()
        print("\nPrelim tf-idf scoring...", end=" ")
        prelim_tfidf = StringMatch(self.trainpool[:100], self.trainpool[:100], 1, 3, "word", 2)
        prelim_tfidf.tokenize()
        prelim_df = prelim_tfidf.match()
        prelim_df = prelim_df[prelim_df["Score"] < 0.999]
        prelim_df = prelim_df[["String1", "String2", "Score"]]
        print("process complete.")
        prelim_df.to_csv(self.datadir + 'outputs/scored_trainpool.csv')
        print("Stored as: scored_trainpool.csv")

    # .
    # .
    # .

    def train(self):
        """
        Takes in labeled train pairs and create model.
        """
        ### YOUR CODE GOES HERE
        pass

    def run(self):
        """
        Runs process from various steps depending on presence of output csvs.
        """
        if not os.path.isfile(self.datadir + "outputs/testmatrix_labeled.csv"):
            self.load_files()
            self.preprocess()
            self.provide_test_matrix()
        else:
            print("Labeled test set identified")
            self.read_test_matrix()
            self.train_match_interface()
            self.train()
        print("\n************************\n")

# --- Instantiate here --- #
TestProgram = MatchProgram()
TestProgram.run()
