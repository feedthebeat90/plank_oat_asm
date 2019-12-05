import numpy as np
import pandas as pd
import random
import os
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import hstack
import time

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
        
        #gets rid of exact matches and randomly shuffles list
        output = list(set(output))
        random.shuffle(output)
        
        #setting aside some terms for testing
        self.test_set = output[:20]
        
        #self.dataset is the whole training set
        self.dataset = output[20:]
        
        split_num = math.ceil(len(self.dataset)/2)
        
        self.batches = []
        
        for i in range(2):
            if split_num*(i+1) < len(output):
                self.batches.append(output[int(i*split_num):int(split_num*(i+1))])
            else:
                self.batches.append(output[int(i*split_num):])
        
            
        self.num_clusters = num_clusters
        self.indicator = ''
        self.clusters = {}
        self.models = {}
    
    #all cluster functions cluster self.data, a list of words
    def ask_about_indicator(self):
        indicator = self.indicator
        while indicator != 'y' and indicator != 'n':
            indicator = input("will you be trying to match many mis-spelled words? (y or n): ")

        if indicator == 'y':
            self.indicator = 'char'
        elif indicator == 'n':
            self.indicator = 'word'
    
    def Cluster(self):
        self.ask_about_indicator()
        
        n_clusters = math.floor(self.num_clusters/2)
        
        if self.indicator == 'word':
            print('Vectorizing by word tfidf...')
            Vectorizer = TfidfVectorizer(ngram_range=(1,1))
            Vectorizer.fit(self.dataset)

        elif self.indicator == 'char':
            print('Vectorizing by character count...')
            Vectorizer = CountVectorizer(ngram_range=(4,4), binary=False, analyzer='char')
            Vectorizer.fit(self.dataset)
        
    
        batch_index = 0
 
        for batch in self.batches[:3]:
            print(len(batch))
            start_time = time.time()
            batch_index += 1
            print('Vectorizing Batch', batch_index, '...')
            features = Vectorizer.transform(batch)

            print('Starting K-means Model for Batch', batch_index, '...')
            k_means = MiniBatchKMeans(n_clusters=n_clusters,batch_size=1000).fit(features)

            clustered_words = pd.DataFrame({'word':batch,'label':k_means.labels_})
            #sorted_clusters = clustered_words.sort_values(['label'])
            print('Storing clusters for Batch', batch_index, '...')
            print('')
            print('Batch', batch_index, 'took ', time.time()-start_time,'seconds')
            self.clusters[batch_index] = clustered_words
            self.models[batch_index] = k_means
        
    
    