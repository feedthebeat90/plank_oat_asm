# Load libraries
import re
import time
import operator

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
import pandas as pd

import sparse_dot_topn.sparse_dot_topn as ct

# A class for matching one list of strings to another
class StringMatch():
    
    def __init__(self, source_names, target_names, ngram_lower, ngram_upper, analyzer, ntop):
        self.source_names = source_names
        self.target_names = target_names
        self.ct_vect      = None
        self.ngram_lower = ngram_lower
        self.ngram_upper = ngram_upper
        self.analyzer = analyzer
        self.ntop = ntop
        self.tfidf_vect   = None
        self.vocab        = None
        self.sprse_mtx    = None
        
        
    def tokenize(self, analyzer='char_wb'):
        '''
        Tokenizes the list of strings, based on the selected analyzer
        :param str analyzer: Type of analyzer ('char_wb', 'word'). Default is trigram
        :param str n: If using n-gram analyzer, the gram length
        '''
        # Create initial count vectorizer & fit it on both lists to get vocab
        self.ct_vect = CountVectorizer(analyzer=analyzer, ngram_range=(self.ngram_lower, self.ngram_upper))
        self.vocab   = self.ct_vect.fit(self.source_names + self.target_names).vocabulary_
        
        # Create tf-idf vectorizer
        self.tfidf_vect  = TfidfVectorizer(vocabulary=self.vocab, analyzer=analyzer, ngram_range=(self.ngram_lower, self.ngram_upper))
        
        
    def match(self, lower_bound=0, output_fmt='df'):
        '''
        Main match function. Default settings return only the top candidate for every source string.
        
        :param int ntop: The number of top-n candidates that should be returned
        :param float lower_bound: The lower-bound threshold for keeping a candidate, between 0-1.
                                   Default set to 0, so consider all canidates
        :param str output_fmt: The output format. Either dataframe ('df') or dict ('dict')
        '''
        self._awesome_cossim_top(self.ntop, lower_bound)
        
        if output_fmt == 'df':
            match_output = self._make_matchdf()
        elif output_fmt == 'dict':
            match_output = self._make_matchdict()
            
        return match_output
        
        
    def _awesome_cossim_top(self, ntop, lower_bound):
        ''' https://gist.github.com/ymwdalex/5c363ddc1af447a9ff0b58ba14828fd6#file-awesome_sparse_dot_top-py '''
        # To CSR Matrix, if needed
        A = self.tfidf_vect.fit_transform(self.source_names).tocsr()
        B = self.tfidf_vect.fit_transform(self.target_names).transpose().tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32

        nnz_max = M * ntop

        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)

        self.sprse_mtx = csr_matrix((data,indices,indptr), shape=(M,N))
    
    
    def _make_matchdf(self):
        ''' Build dataframe for result return '''
        # CSR matrix -> COO matrix
        cx = self.sprse_mtx.tocoo()

        # COO matrix to list of tuples
        match_list = []
        for row,col,val in zip(cx.row, cx.col, cx.data):
            match_list.append((row, self.source_names[row], col, self.target_names[col], val))

        # List of tuples to dataframe
        colnames = ['Row Idx', 'String1', 'Candidate Idx', 'String2', 'Score']
        match_df = pd.DataFrame(match_list, columns=colnames)

        return match_df

    
    def _make_matchdict(self):
        ''' Build dictionary for result return '''
        # CSR matrix -> COO matrix
        cx = self.sprse_mtx.tocoo()

        # dict value should be tuple of values
        match_dict = {}
        for row,col,val in zip(cx.row, cx.col, cx.data):
            if match_dict.get(row):
                match_dict[row].append((col,val))
            else:
                match_dict[row] = [(col, val)]

        return match_dict


# source_titles = pd.read_csv('csvs/bonica_orgs_reduced.csv')['x'].tolist()[:100]
# sourcedf = pd.DataFrame(source_titles, columns=['str1'])
# sourcedf.to_csv('csvs/mini_bonica.csv', index=False)
# target_titles = pd.read_csv('csvs/amicus_org_names.csv')['x'].tolist()

# titlematch = StringMatch(source_titles, target_titles, 1, 3, "word")
# titlematch.tokenize()
# match_df = titlematch.match()
# match_df.head()
# match_df.to_csv('csvs/tf_idf_test_out.csv')