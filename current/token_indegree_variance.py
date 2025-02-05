
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from encoding_similarity import get_index
from heatmaps import make_heatmap
from configure import NLLB_SEED_LANGS, SEED_EMBED_PICKLE, TEN_SEED_LANGS


def find_node_indegrees(embedding_matrix, faiss_index):
    query_vector = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    _, indices = faiss_index.search(query_vector, 1)

    indices = [i[0] for i in indices]
    d = {}
    for index in indices:
        if index in d:
            d[index] += 1
        else:
            d[index] = 1

    indegrees = list(d.values())
    if len(indegrees) > 1:
        return np.var(indegrees, ddof=1)

    return None

def token_pair_similarity(data, lang1, lang2, sent_id, verbose=False):
    """Computes the average max similarity for the sentence tokens."""
    l1_embeddings = data[(lang1, sent_id)]
    l2_embeddings = data[(lang2, sent_id)]
    l1_query_vector = l1_embeddings[1:-1].astype('float32') # exclude language tag and end of sentence token
    l2_query_vector = l2_embeddings[1:-1].astype('float32') # exclude language tag and end of sentence token

    varAB = find_node_indegrees(l1_query_vector, get_index(data, lang2, sent_id))
    varBA = find_node_indegrees(l2_query_vector, get_index(data, lang1, sent_id))

    return varAB, varBA

def main(token_indegree, length_var):
    config_file = sys.argv[1] 
    with open(config_file) as reader:
        config = json.load(reader)

    exp_dir = config['experiment_directory']
    if not os.path.exists(exp_dir):
        raise Exception(f'Experiment directory does not exist: {exp_dir}')
    
    save_dir = os.path.join(exp_dir, 'token_indegree')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('loading embeddings...')
    df = pd.read_pickle(config['parallel_corpus_pkl'])
    languages = config['languages']
    range_start, range_end = config['sentence_range']

    df = df[(df.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= df['sent_id']) & (df['sent_id'] <= range_end)]

    var_table = np.full((len(languages), len(languages)), 0.0)

    print('compiling embeddings...')
    data = {}
    # make dict for speed & cleanliness
    for _, row in df.iterrows():
        language = f"{row['language']}_{row['script']}"
        id = row['sent_id']
        data[(language, id)] = row['embedding']

    if length_var:
        print('computing length variances...')
        vars = []
        # for id in tqdm(range(range_start, range_end)):
        for id in range(range_start, range_end):
            lens = [len(data[(languages[i], id)]) for i in range(0, len(languages))]
            vars.append(np.var(lens, ddof=1))
        avg_stdev = np.mean([np.sqrt(var) for var in vars])
        avg_var = np.mean(vars)
        print('average variance', avg_var)
        print('average standev', avg_stdev)
            

    if token_indegree:
        print('computing indegree variances...')
        for i in range(0, len(languages)):
            print(languages[i])
            for j in tqdm(range(i + 1, len(languages))):
                vars12 = []
                vars21 = []
                lang1=languages[i]
                lang2=languages[j]
                print(lang1, lang2)
                for id in range(range_start, range_end):   
                    var12, var21 = token_pair_similarity(data, lang1, lang2, id, verbose=False)
                    if var12 is not None and var21 is not None:
                        vars12.append(var12)
                        vars21.append(var21)
                var_table[i][j] = np.mean(vars12)
                var_table[j][i] = np.mean(vars21)

        idx_pairs = [(i, j) for i in range(0, len(languages)) for j in range(0, len(languages)) if i != j]
        record = [{'lang1': languages[i], 'lang2': languages[j], 'variance': var_table[i][j]} for (i, j) in idx_pairs]
        record_df = pd.DataFrame(record)
        record_df.to_csv(os.path.join(save_dir, 'token_indegree_variances.csv'), index=False)

        make_heatmap(var_table, "var_unordered", save_dir, languages)
        make_heatmap(var_table, "var_clustered", save_dir, languages, cluster=True)


if __name__ == "__main__":
    main(False, True)

""""
seed:
average variance 223.730
average standev 14.417

europarl:
average variance 42.810
average standev 5.890

"""