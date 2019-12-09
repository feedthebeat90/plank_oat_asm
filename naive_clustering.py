def read_data(*argv):
    """
    takes in a list of csv paths and concatenates them
    """
    print("\n***** ClusterProgram *****\n")

    #concatenates all datasets into one long list
    output = []
    for file_path in argv:
        df = pd.read_csv(file_path)
        df.columns = ['id', 'words']
        output += list(df['words'])

    #gets rid of exact matches, changes all to lower case, and removes characters
    print('preprocessing terms...')
    output = list(set(output))
    output = [x.lower() for x in output]
    to_be_removed = [",", "'", '"', '-', '_','.']
    for char in to_be_removed:
        output = [x.replace(char,'') for x in output]
    output = [x.replace('&', 'and') for x in output]
        
    return output

def build_vocab(dataset):
    word_count = {}
    for term in dataset:
        for word in term.split(' '):
            if word in word_count.keys():
                word_count[word] += 1
            else:
                word_count[word] = 1
    word_count = {k:v for k,v in word_count.items() if v != 1}
    sorted_words = sorted(word_count, key=word_count.get)
    return word_count, sorted_words

def find_and_remove(word, dataset):
    cluster = []
    copy = []
    for term in dataset:
        if word in term:
            cluster.append(term)
        else:
            copy.append(term)
    return cluster, copy
            


def cluster(dataset, sorted_words):
    clusters = {}
    to_be_shortened = []
    for term in dataset:
        to_be_shortened.append(term)
    count = 0
    for word in sorted_words:
        if len(to_be_shortened) > 0:
            if count%500 == 0:
                print(count, word)
            count += 1
            cluster, to_be_shortened = find_and_remove(word, to_be_shortened)
            clusters[count] = cluster
            
    for term in to_be_shortened:
        count += 1
        clusters[count] = [term]
            
    return clusters

def get_cluster(term, clusters):
    for key in clusters.keys():
        if term in clusters[key]:
            return clusters[key]
    return None

def evaluate(test_set_path, clusters):
    fn = 0
    fp = 0
    tn = 0
    tp = 0
    
    
    test_df = pd.read_csv(test_set_path)
    print(test_df.columns)
    if 'term' in test_df.columns:
        test_df = test_df.drop(['Unnamed: 0', 'term'], axis=1)
    else:
        test_df = test_df.drop(['Unnamed: 0'], axis=1)
    test_terms = test_df.columns
    test_terms = [x.lower() for x in test_terms]
    to_be_removed = [",", "'", '"', '-', '_','.']
    for char in to_be_removed:
        test_terms = [x.replace(char,'') for x in test_terms]
    test_terms = [x.replace('&', 'and') for x in test_terms]
    
    test_matrix = np.array(test_df)
    
    for i in range(test_matrix.shape[0]):
        term1 = test_terms[i]
        
        i_terms = get_cluster(term1, clusters)
        if not i_terms is None:
            
            for j in range(test_matrix.shape[1]):
                if i > j:
                    true_match_indicator = test_matrix[i][j]
                    term2 = test_terms[j]
                    predicted_match_indicator = 1*(term2 in i_terms)

                    if true_match_indicator == 0 and predicted_match_indicator == 0:
                        tn += 1
                    elif true_match_indicator == 0 and predicted_match_indicator == 1:
                        fp += 1
                        print('')
                        print('false positive example:')
                        print('term 1:', term1, 'term 2:', term2)
                        print('')
                    elif true_match_indicator == 1 and predicted_match_indicator == 0:
                        #print('')
                        #print('false negative example:')
                        #print('term 1:', term1, 'term 2:', term2)
                        #print('')
                        fn += 1
                    elif true_match_indicator == 1 and predicted_match_indicator == 1:
                        print('')
                        print('true positive example:')
                        print('term 1:', term1, 'term 2:', term2)
                        print('')
                        tp += 1

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    print('tp:', tp, 'fn:',fn)
                
    return precision, recall

def print_random_matches(clusters, num_matches):
    matches = []
    keys = np.random.choice(list(clusters.keys()), size=num_matches, replace=True)
    for key in keys:
        cluster = clusters[key]
        if len(cluster) == 0:
            do_nothing = True
        elif len(cluster) == 1:
            matches.append((cluster[0], 'no match in dataset'))
        elif len(cluster) == 2:
            matches.append((cluster[0], cluster[1]))
        else:
            matches.append(tuple(np.random.choice(cluster, size=2, replace=False)))
                           
    return matches
                           
print_random_matches(clusters, 50)

dataset = read_data('csvs/amicus_org_names.csv', 'csvs/bonica_org_names.csv')
dataset_vocab, sorted_words = build_vocab(dataset)
preserved = dataset
clusters = cluster(preserved, sorted_words)
print('Results For Representative Test Set')
precision,recall = evaluate('csvs/outputs/testmatrix_labeled.csv', clusters)
print('')
print('##################################')
print('precision:', precision)
print('recall:', recall)
print('')
print('Results For Match Dense Test Set')
precision,recall = evaluate('csvs/handcoded_test.csv.csv', clusters)
print('')
print('##################################')
print('precision:', precision)
print('recall:', recall)

print_random_matches(clusters, 50)