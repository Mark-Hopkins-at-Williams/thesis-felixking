import sys
sys.path.append('/mnt/storage/fking/thesis-felixking/finetuning')

import torch
import faiss
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from finetune import tokenize
import torch.nn.functional as F
from heatmaps import make_heatmap
from bottle import CustomM2M100Model
from scipy.spatial.distance import cosine
from configure import NLLB_SEED_CSV, NLLB_SEED_LANGS, SEED_EMBED_PICKLE, TEN_SEED_LANGS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

def make_index(data, lang, sent_id):
    lang_data = (data[(lang, sent_id)][0][1:-1], data[(lang, sent_id)][1][1:-1])
    embeddings = lang_data[0]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.float32(embeddings))
    faiss.write_index(index, f'indices/{lang}_{sent_id}')

def token_pair_similarity(data, lang1, lang2, sent_id, verbose=False):


    l1_data = data[(lang1, sent_id)][0][1:-1]
    l2_data = data[(lang2, sent_id)][0][1:-1]

    # token embeddings
    query_vector = l1_data.astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
    index = faiss.read_index(f'indices/{lang2}_{sent_id}')
    
    distances, _ = index.search(query_vector, 1)
    distancesAB = [a[0] for a in distances]j

    query_vector = l2_data.astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
    index = faiss.read_index(f'indices/{lang1}_{sent_id}')
    
    distances, _ = index.search(query_vector, 1)
    distancesBA = [a[0] for a in distances]

    # could act as a kind of score for a language pair
    combined_mean = np.mean(distancesAB + distancesBA)

    if verbose:
        print(f'\n{lang1}, {lang2}: ', combined_mean)

    return combined_mean

def main():
    
    df = pd.read_pickle(SEED_EMBED_PICKLE) # had to do a pickle file :^|
    
    # languages = TEN_SEED_LANGS
    languages = NLLB_SEED_LANGS
    sentence_range = range(0, df['sent_id'].max())
    # sentence_range = range(0, 100)

    score_table = np.full((len(languages), len(languages)), 0.5)

    data = {}
    # make dict for speed & cleanliness
    for index, row in df.iterrows():
        language = f"{row['language']}_{row['script']}"
        id = row['sent_id']
        data[(language, id)] = (row['embedding'], row['tokens'])

    scores = []
    start = time.perf_counter()

    with open('record.txt', 'a') as file:
        for i in range(0, len(languages)):
            for j in range(i + 1, len(languages)):

                lp_scores = []
                lang1=languages[i]
                lang2=languages[j]

                for id in sentence_range:
                    
                    score = token_pair_similarity(data, lang1, lang2, id, verbose=False)
                    lp_scores.append(score)

                file.write(f'\n{lang1}, {lang2}: {np.mean(lp_scores)}')

                score_table[i][j] = np.mean(lp_scores)
                score_table[j][i] = np.mean(lp_scores)

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")

    make_heatmap(score_table, "test_heat", '../plots/heatmaps', languages)
    make_heatmap(score_table, "test_cluster", '../plots/heatmaps', languages, cluster=True)

if __name__ == "__main__":
    main()
