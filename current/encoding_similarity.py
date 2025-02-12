
import os
import sys
import json
import faiss # type: ignore
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from heatmaps import make_heatmap

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


def token_pair_similarity(data, lang1, lang2, sent_id, verbose=False, geometric_mean=False):
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

def main():
    config_file = sys.argv[1] 
    with open(config_file) as reader:
        config = json.load(reader)

    exp_dir = config['experiment_directory']
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)


    print('loading embeddings...')
    df = pd.read_pickle(config['parallel_corpus_pkl'])
    languages = config['languages']
    range_start, range_end = config['sentence_range']
    primary_lang = config['primary']
    primary_sim_col = primary_lang.split('_')[0] + '_sim'
    secondary_lang = config['secondary']
    secondary_sim_col = secondary_lang.split('_')[0] + '_sim'

    df = df[(df.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= df['sent_id']) & (df['sent_id'] <= range_end)]

    for model_size in ['600M', '1.3B']:

        save_dir = os.path.join(exp_dir, model_size)
        os.mkdir(save_dir)

        score_table = np.full((len(languages), len(languages)), 1.0)

        print('compiling embeddings...')
        data = {}
        # make dict for speed & cleanliness
        for _, row in df.iterrows():
            language = f"{row['language']}_{row['script']}"
            id = row['sent_id']
            data[(language, id)] = row[f'{model_size}_embedding']

        avgs = [{} for l in languages]
        save_lines = []
        print('computing similarities...')
        for i in tqdm(range(0, len(languages))):
            avgs[i]['language'] = languages[i]
            for j in range(i + 1, len(languages)):
                lp_scores = []
                lang1=languages[i]
                lang2=languages[j]
                for id in range(range_start, range_end):   
                    score = token_pair_similarity(data, lang1, lang2, id, verbose=False)
                    lp_scores.append(score)
                mean = np.mean(lp_scores)
                score_table[i][j] = mean
                score_table[j][i] = mean
                save_lines.append(f'{lang1}, {lang2}: {mean:.4f}')
                if lang2 == primary_lang:         # capture similarities to eng and ces
                    avgs[i][primary_sim_col] = mean   # for plotting later
                elif lang1 == primary_lang:
                    avgs[i][primary_sim_col] = 1.0
                    avgs[j][primary_sim_col] = mean
                if lang2 == secondary_lang:
                    avgs[i][secondary_sim_col] = mean
                elif lang1 == secondary_lang:
                    avgs[i][secondary_sim_col] = 1.0
                    avgs[j][secondary_sim_col] = mean

            avgs[i]['avg_sim'] = np.mean(score_table[i]) # also get avg similarity
                
        with open(os.path.join(save_dir, 'data.txt')) as file:
            file.write('\n'.join(save_lines))
            
        scores = pd.DataFrame(avgs)
        scores.to_csv(os.path.join(save_dir, 'similarities.csv'), index=False)

        make_heatmap(score_table, "unordered", save_dir, languages)
        make_heatmap(score_table, "clustered", save_dir, languages, cluster=True)

    shutil.copy(config_file, os.path.join(exp_dir, os.path.basename(config_file)))

if __name__ == "__main__":
    main()
