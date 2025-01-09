
import faiss
import json
import numpy as np
import pandas as pd
from heatmaps import make_heatmap
from configure import NLLB_SEED_LANGS, SEED_EMBED_PICKLE, TEN_SEED_LANGS
import os
import shutil
import sys
from tqdm import tqdm


def find_closest_distances(embedding_matrix, lang, sent_id):
    query_vector = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    faiss_index = faiss.read_index(f'indices/{lang}_{sent_id}')
    distances, _ = faiss_index.search(query_vector, 1)
    return [a[0] for a in distances] # the closest distance is the zeroth element of each list



def token_pair_similarity(data, lang1, lang2, sent_id, verbose=False):
    """Computes the average max similarity for the sentence tokens."""
    l1_embeddings, _ = data[(lang1, sent_id)]
    l2_embeddings, _ = data[(lang1, sent_id)]
    l1_query_vector = l1_embeddings[1:-1].astype('float32') # exclude language tag and end of sentence token
    l2_query_vector = l2_embeddings[1:-1].astype('float32') # exclude language tag and end of sentence token
    distancesAB = find_closest_distances(l1_query_vector, lang2, sent_id)
    distancesBA = find_closest_distances(l2_query_vector, lang1, sent_id)
    combined_mean = np.mean(distancesAB + distancesBA) # average all bidirectional distances
    if verbose: # print result if desired
        print(f'\n{lang1}, {lang2}: ', combined_mean)
    return combined_mean

def main():
    config_file = sys.argv[1] 
    with open(config_file) as reader:
        config = json.load(reader)

    exp_dir = config['experiment_directory']
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    else:
        raise Exception(f'Experiment directory already exists: {exp_dir}')


    print('loading embeddings...')
    df = pd.read_pickle(config['parallel_corpus_file']) # had to do a pickle file :^|
    
    languages = config['languages']
    range_start, range_end = config['sentence_range']

    score_table = np.full((len(languages), len(languages)), 1.0)

    print('compiling embeddings...')
    data = {}
    # make dict for speed & cleanliness
    for _, row in df.iterrows():
        language = f"{row['language']}_{row['script']}"
        id = row['sent_id']
        data[(language, id)] = (row['embedding'], row['tokens'])


    print('computing similarities...')
    with open(os.path.join(exp_dir, 'similarities.txt'), 'w') as file:
        for i in tqdm(range(0, len(languages))):
            for j in range(i + 1, len(languages)):
                lp_scores = []
                lang1=languages[i]
                lang2=languages[j]
                for id in range(range_start, range_end):   
                    score = token_pair_similarity(data, lang1, lang2, id, verbose=False)
                    lp_scores.append(score)
                file.write(f'\n{lang1}, {lang2}: {np.mean(lp_scores)}')
                score_table[i][j] = np.mean(lp_scores)
                score_table[j][i] = np.mean(lp_scores)


    make_heatmap(score_table, "unordered", exp_dir, languages)
    make_heatmap(score_table, "clustered", exp_dir, languages, cluster=True)

    shutil.copy(config_file, os.path.join(exp_dir, os.path.basename(config_file)))

if __name__ == "__main__":
    main()
