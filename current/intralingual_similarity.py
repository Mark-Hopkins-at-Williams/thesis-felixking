
import os
import sys
import json
import faiss # type: ignore
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt # type: ignore

def plot_dict_to_bar(data_dict, filename, figsize=(10,6), title=None, xlabel=None, ylabel=None):
   plt.figure(figsize=figsize)
   plt.bar(data_dict.keys(), data_dict.values())
   if title: plt.title(title)
   if xlabel: plt.xlabel(xlabel)
   if ylabel: plt.ylabel(ylabel)
   plt.xticks(rotation=45, ha='right')
   plt.tight_layout()
   plt.savefig(filename)
   plt.close()

def get_index(data, lang, sent_id, sent_set):
    embeddings = data[(lang, sent_id, sent_set)][1:-1]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.float32(embeddings))
    return index

def run_faiss(embedding_matrix, index):
    query_vector = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    distances, _ = index.search(query_vector, 1)
    return [a[0] for a in distances] # the closest distance is the zeroth element of each list

def token_pair_similarity(data, language, id):
    embedsA = data[(language, id, 'a')]
    embedsB = data[(language, id, 'b')]

    qvA = embedsA[1:-1].astype('float32') 
    ixB = get_index(data, language, id, 'b')

    qvB = embedsB[1:-1].astype('float32') 
    ixA = get_index(data, language, id, 'a')

    distancesAB = run_faiss(qvA, ixB)
    distancesBA = run_faiss(qvB, ixA)

    print(np.mean(distancesAB), np.mean(distancesBA))

    return np.sqrt(np.mean(distancesAB) * np.mean(distancesBA)) # geometric mean


def old(data, language, id1, id2, query_vecs, indices):
    """Computes the average max similarity for the sentence tokens."""

    if (language, id1) not in query_vecs:
        embeddings = data[(language, id1)]
        query_vec = embeddings[1:-1].astype('float32') # exclude language tag and end of sentence token
        index = get_index(data, language, id1)
        query_vecs[(language, id1)] = query_vec
        indices[(language, id1)] = index
    
    if (language, id2) not in query_vecs:
        embeddings = data[(language, id2)]
        query_vec = embeddings[1:-1].astype('float32') # exclude language tag and end of sentence token
        index = get_index(data, language, id2)
        query_vecs[(language, id2)] = query_vec
        indices[(language, id2)] = index
    
    qv1 = query_vecs[(language, id1)]
    index1 = indices[(language, id1)]
    qv2 = query_vecs[(language, id2)]
    index2 = indices[(language, id2)]

    distancesAB = run_faiss(qv1, index2)
    distancesBA = run_faiss(qv2, index1)

    return np.sqrt(np.mean(distancesAB) * np.mean(distancesBA)) # geometric mean


def main():
    config_file = sys.argv[1] 
    with open(config_file) as reader:
        config = json.load(reader)

    exp_dir = config['experiment_directory']
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    languages = config['languages']
    range_start, range_end = config['sentence_range']
    
    print('loading embeddings...')
    dfA = pd.read_pickle(config['intralingual_A_pkl'])
    dfB = pd.read_pickle(config['intralingual_B_pkl'])
    dfA = dfA[(dfA.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= dfA['sent_id']) & (dfA['sent_id'] <= range_end)]
    dfB = dfB[(dfA.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= dfA['sent_id']) & (dfA['sent_id'] <= range_end)]

    for model_size in ['600M', '1.3B']:

        save_dir = os.path.join(exp_dir, model_size)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        print('compiling embeddings...')
        print(model_size)
        data = {}
        # make dict for speed & cleanliness

        for i in range(len(dfA)):
            row = dfA.iloc[i]  # Get row by position
            language = f"{row['language']}_{row['script']}"
            id = row['sent_id']
            sent_set = row['set']
            
            data[(language, id, sent_set)] = row[f'{model_size}_embedding']
        
        for i in range(len(dfB)):
            row = dfB.iloc[i]  # Get row by position
            language = f"{row['language']}_{row['script']}"
            id = row['sent_id']
            sent_set = row['set']

            data[(language, id, sent_set)] = row[f'{model_size}_embedding']

        scores = {}
        print('computing similarities...')
        with open(os.path.join(save_dir, 'intralingual_similarities.txt'), 'w') as file:
            for language in tqdm(languages):
                lang_scores = []
                for id in range(range_start, range_end):
                    lang_scores.append(token_pair_similarity(data, language, id))
                mean = np.mean(lang_scores)
                scores[language] = mean
                file.write(f'{language}: {mean}\n')
                
        plot_dict_to_bar(
            scores,
            os.path.join(save_dir, 'intralingual.png'),
            title='Average Intralingual Encoding Similarity',
            xlabel='Language',
            ylabel='Average Encoding Similarity',
        )

            
if __name__ == "__main__":
    main()
