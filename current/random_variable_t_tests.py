import os
import sys
import json
import faiss # type: ignore
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from heatmaps import make_heatmap # type: ignore

MODEL_SIZE = '600M'
NUM_PAIRS = 10000

def get_index(data, lang, sent_id):
    embeddings = data[(lang, sent_id)][1:-1]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.float32(embeddings))
    return index

def find_closest_distances(embedding_matrix, lang, sent_id, index=None):
    query_vector = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    if index:
        faiss_index = index
    else:
        faiss_index = faiss.read_index(f'indices/{lang}_{sent_id}')
    distances, _ = faiss_index.search(query_vector, 1)
    return [a[0] for a in distances] # the closest distance is the zeroth element of each list

def token_pair_similarity(data, lang1, lang2, sent_id, geometric_mean=True):
    """Computes the average max similarity for the sentence tokens."""
    l1_embeddings = data[(lang1, sent_id)]
    l2_embeddings = data[(lang2, sent_id)]
    l1_query_vector = l1_embeddings[1:-1].astype('float32') # exclude language tag and end of sentence token
    l2_query_vector = l2_embeddings[1:-1].astype('float32') # exclude language tag and end of sentence token
    
    distancesAB = find_closest_distances(l1_query_vector, lang2, sent_id, index=get_index(data, lang2, sent_id))
    distancesBA = find_closest_distances(l2_query_vector, lang1, sent_id, index=get_index(data, lang1, sent_id))

    if geometric_mean:
        return np.sqrt(np.mean(distancesAB) * np.mean(distancesBA))
    else:
        return np.mean(distancesAB + distancesBA) # average all bidirectional distances

######### INTRA ###########

def intra_get_index(data, lang, sent_id, sent_set):
    embeddings = data[(lang, sent_id, sent_set)][1:-1]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.float32(embeddings))
    return index

def run_faiss(embedding_matrix, index):
    query_vector = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    distances, _ = index.search(query_vector, 1)
    return [a[0] for a in distances] # the closest distance is the zeroth element of each list

def intra_token_pair_similarity(data, language, id):
    embedsA = data[(language, id, 'a')]
    embedsB = data[(language, id, 'b')]

    qvA = embedsA[1:-1].astype('float32') 
    ixB = intra_get_index(data, language, id, 'b')

    qvB = embedsB[1:-1].astype('float32') 
    ixA = intra_get_index(data, language, id, 'a')

    distancesAB = run_faiss(qvA, ixB)
    distancesBA = run_faiss(qvB, ixA)

    # print(np.mean(distancesAB), np.mean(distancesBA))

    return np.sqrt(np.mean(distancesAB) * np.mean(distancesBA)) # geometric mean

def get_interlingual(config):

    exp_dir = config['experiment_directory']
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    print('loading embeddings...')
    df = pd.read_pickle(config['parallel_corpus_pkl'])
    languages = config['languages']
    range_start, range_end = config['sentence_range']
    df = df[(df.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= df['sent_id']) & (df['sent_id'] <= range_end)]

    save_dir = os.path.join(exp_dir, MODEL_SIZE)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('compiling embeddings...')
    data = {}
    for _, row in df.iterrows():
        language = f"{row['language']}_{row['script']}"
        id = row['sent_id']
        data[(language, id)] = row[f'{MODEL_SIZE}_embedding']

    scores = []
    print('computing similarities...')
    for i in tqdm(range(0, len(languages))):
        for j in range(i + 1, len(languages)):
            lang1, lang2 = languages[i], languages[j]
            for id in range(range_start, range_end):   
                scores.append(token_pair_similarity(data, lang1, lang2, id))
    
    return scores
        
def get_intralingual(config):

    exp_dir = config['experiment_directory']
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    languages = config['languages']
    range_start, range_end = config['sentence_range']
    
    print('loading embeddings...')
    dfA = pd.read_pickle(config['intralingual_A_pkl'])
    dfB = pd.read_pickle(config['intralingual_B_pkl'])
    dfA = dfA[(dfA.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= dfA['sent_id']) & (dfA['sent_id'] <= range_end)]
    dfB = dfB[(dfB.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= dfB['sent_id']) & (dfB['sent_id'] <= range_end)]

    save_dir = os.path.join(exp_dir, MODEL_SIZE)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('compiling embeddings...')
    data = {}
    for i in range(len(dfA)):
        row = dfA.iloc[i]  # Get row by position
        language = f"{row['language']}_{row['script']}"
        id = row['sent_id']
        sent_set = row['set']
        
        data[(language, id, sent_set)] = row[f'{MODEL_SIZE}_embedding']
    
    for i in range(len(dfB)):
        row = dfB.iloc[i]  # Get row by position
        language = f"{row['language']}_{row['script']}"
        id = row['sent_id']
        sent_set = row['set']

        data[(language, id, sent_set)] = row[f'{MODEL_SIZE}_embedding']

    print('computing similarities...')
    for language in tqdm(languages):
        scores = []
        for id in range(range_start, range_end):
            scores.append(intra_token_pair_similarity(data, language, id))

    return scores

def get_spurious(config):
    return get_interlingual(config)

def save_data(name, config, data):
    with open(os.path.join(config['experiment_directory'], f'{MODEL_SIZE}/{name}_sims.txt'), 'w') as file:
        file.write('\n'.join([str(x) for x in data]))

def get_data(exp):
    normal_config_file = f'exp_configs/{exp}.json'
    scrambled_config_file = f'exp_configs/scrambled_{exp}.json'

    with open(normal_config_file) as reader:
        normal_config = json.load(reader)
    
    with open(scrambled_config_file) as reader:
        scrambled_config = json.load(reader)

    print('~INTERLINGUAL~')
    interlingual_similarities = get_interlingual(normal_config)
    print('~INTRALINGUAL~')
    intralingual_similarities = get_intralingual(normal_config)
    print('~SPURIOUS~')
    spurious_similarities = get_spurious(scrambled_config)

    save_data('interlingual', normal_config, interlingual_similarities)
    save_data('intralingual', normal_config, intralingual_similarities)
    save_data('spurious', normal_config, spurious_similarities)

    return interlingual_similarities, intralingual_similarities, spurious_similarities

def calculate_expected_val(first, second):

    f = np.array(random.choices(first, k=NUM_PAIRS))
    s = np.array(random.choices(second, k=NUM_PAIRS))
    return np.sum(f>s)
    
if __name__ == "__main__":
    exp = ''
    if len(sys.argv) == 2:
        exp = sys.argv[1] 
    else:
        print('usage: python random_variable_t_tests.py <dataset_name>')
        print('quitting...')
        exit()

    inter, intra, spur = get_data(exp)

    rv1 = calculate_expected_val(inter, intra)
    rv2 = calculate_expected_val(inter, spur)
    rv3 = calculate_expected_val(intra, spur)

    print(f'rv1: {rv1:.3f}')
    print(f'rv2: {rv2:.3f}')
    print(f'rv3: {rv3:.3f}')

###################################################################################################
# 
# For random variable t-testing, need to take a bunch of pairs of instances of the random 
# variables, that is, thousands of interlingual, intralingual, and spurious similarities.
#
# Then, repeatedly compare and keep track of which is higher. 
#
# for example, use 10,000 interlingual, intralingual, and spurious pairs from Europarl
# calculate all these similarities and store them in 3 lists.
# then randomly sample, say, 10,000 from each one (with repeats)
# one by one, compare lists A and B, adding 1 to a sum if A[i] > B[i], then divide by 10,000
# 
# Do this for all three pairs: (A,B) (A,C), and (B,C)
# 
# Null hypotheses would be that these three numbers are all 0.5s
# 
###################################################################################################