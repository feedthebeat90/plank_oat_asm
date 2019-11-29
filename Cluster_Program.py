import numpy as np
import pandas as pd
import random
import os
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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
            
        self.data = output
            
    def strip_file(self,*argv):
        output = []
        for file_path in argv:
            output += list(pd.read_csv(file_path))
            
        return output
    
    #all cluster functions cluster self.data, a list of words
    
    def Tfidf_word_cluster(self, num_clusters):
        print('Clustering by words and Tfidf...')
        Tfidf_Vectorizer = TfidfVectorizer(ngram_range=(1,3), binary=True)
        Tfidf_Vectorizer.fit(self.dataset)

        vectorized_words = Tfidf_Vectorizer.transform(self.dataset)

        k_means = KMeans(n_clusters = self.num_clusters,init='k-means++', precompute_distances='auto').fit(vectorized_words)

        clustered_words = pd.DataFrame({'word':test_df['words'],'label':k_means.labels_})
        sorted_clusters = clustered_words.sort_values(['label'])

        return sorted_clusters
    
    def count_word_cluster(self, num_clusters):
        print('Clustering by words and Count...')
        Count_Vectorizer = CountVectorizer(ngram_range=(1,3), binary=True)
        Count_Vectorizer.fit(self.dataset)

        vectorized_words = Count_Vectorizer.transform(self.dataset)

        k_means = KMeans(n_clusters = self.num_clusters,init='k-means++', precompute_distances='auto').fit(vectorized_words)

        clustered_words = pd.DataFrame({'word':test_df['words'],'label':k_means.labels_})
        sorted_clusters = clustered_words.sort_values(['label'])

        return sorted_clusters
    
    #useful for mispelled words
    def count_char_cluster(self, num_clusters):
        print('Clustering by characters and Count...')
        Count_Vectorizer = CountVectorizer(ngram_range=(4,8), binary=False, analyzer='char')
        Count_Vectorizer.fit(self.dataset)

        vectorized_words = Count_Vectorizer.transform(self.dataset)

        k_means = KMeans(n_clusters = self.num_clusters,init='k-means++', precompute_distances='auto').fit(vectorized_words)

        clustered_words = pd.DataFrame({'word':test_df['words'],'label':k_means.labels_})
        sorted_clusters = clustered_words.sort_values(['label'])

        return sorted_clusters
    
    def same_cluster(self, tfidf_cluster, char_cluster):
        count = 0
        for word in char_cluster:
            if word in tfidf_cluster:
                count += 1   
        return count/len(char_cluster) > .7


    def combine_clusters(self, tfidf_clusters, char_clusters):
        output = {}
        for i in range(self.num_clusters):
            output[i] = []
            tfidf_label = tfidf_clusters.loc[tfidf_clusters['label'] == i]
            for j in range(self.num_clusters):
                char_label = char_clusters.loc[char_clusters['label'] == j]
                if same_cluster(list(tfidf_label['word']), list(char_label['word'])):
                    #print(list(tfidf_label['word']), list(char_label['word']))
                    output[i] += list(set(list(tfidf_label['word']) + list(char_label['word'])))
                else:
                    output[i] += list(tfidf_label['word'])

        return output