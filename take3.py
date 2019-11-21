# --- Add imports here --- #
import numpy as np
import pandas as pd
import random
import os
import io

class MatchProgram():

    def __init__(self, datadir):
        print("\n***** MatchProgram *****\n")
        self.datadir = datadir
        if not os.path.exists(self.datadir + "outputs/"):
            os.makedirs(self.datadir + "outputs/")
    
    def strip_file(self, fn):
        df = pd.read_csv(fn).select_dtypes(include="object")
        if len(df.columns) != 1:
            print("Make sure each file contains 1 column.")
        return [list(set(df[col].tolist())) for col in df][0]

    def load_files(self):
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

    def preprocess(self):
        print("\nLen processed corpus: ")
        print("Stored as: processed.csv")
        # return self.processed
        pass

    def grab_alph_chunk(self, l, n):
        start_idx = random.randint(0, len(l)-n-1)
        return sorted(l)[start_idx:start_idx+n]

    # once preprocess is written, 
    # replace refs to self.unprocessed with self.processed
    def provide_test_matrix(self):
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
        tmp_processed.to_csv(self.datadir + "outputs/trainpool.csv", index=False)
        testmatrix = np.empty((self.testsize, self.testsize))
        testmatrix[:] = np.nan
        testdf = pd.DataFrame(testmatrix, index=self.teststrs, columns=self.teststrs)
        testdf.to_csv(self.datadir + "outputs/testmatrix_unlabeled.csv")
        print("Empty test set generated")
        print("Stored as: testmatrix_unlabeled.csv")
        print("\nPlease label matches as 1, nonmatches as 0")
        print("When you have labeled the full test set, rename file testmatrix_labeled.csv")

    def read_test_matrix(self):
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
        pass

    def run(self):
        if not os.path.isfile(self.datadir + "outputs/testmatrix_labeled.csv"):
            self.load_files()
            self.preprocess()
            self.provide_test_matrix()
            self.run()
        else:
            print("Labeled test set identified")
            self.read_test_matrix()
            self.train_match_interface()
        print("\n************************\n")

# --- Instantiate here --- #
TestProgram = MatchProgram(datadir="csvs/")
TestProgram.run()