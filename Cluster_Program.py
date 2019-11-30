import numpy as np
import pandas as pd
import random
import os
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import hstack

class ClusterProgram():
    
    def __init__(self, num_clusters, *argv):
        """
        takes in a list of csv paths and concatenates them
        """
        print("\n***** ClusterProgram *****\n")
        output = []
        for file_path in argv:
            df = pd.read_csv(file_path)
            df.columns = ['id', 'words']
            output += list(df['words'])
            
        self.dataset = output
        self.num_clusters = num_clusters
    
    #all cluster functions cluster self.data, a list of words
    
    def Tfidf_word_cluster(self):
        print('Vectorizing by word tfidf...')
        Tfidf_Vectorizer = TfidfVectorizer(ngram_range=(0,1), binary=True)
        Tfidf_Vectorizer.fit(self.dataset)

        print('Vectorizing by character count...')
        Count_Vectorizer = CountVectorizer(ngram_range=(4,4), binary=False, analyzer='char')
        Count_Vectorizer.fit(self.dataset)

        Tfidf_words = 2*Tfidf_Vectorizer.transform(self.dataset)
        count_chars = Count_Vectorizer.transform(self.dataset)
        
        print('Concatenating word and character vectors')
        all_features = hstack((Tfidf_words, count_chars))

        print('Starting K-means Model')
        k_means = KMeans(n_clusters = self.num_clusters,init='k-means++', n_init=10, precompute_distances='auto').fit(all_features)

        clustered_words = pd.DataFrame({'word':test_df['words'],'label':k_means.labels_})
        sorted_clusters = clustered_words.sort_values(['label'])
        
        return sorted_clusters