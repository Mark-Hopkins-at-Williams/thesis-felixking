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
    embeddings = data[(lang1, sent_id)][0][1:-1]
    embeddings = np.linalg.norm(data, axis=1, keepdims=True)
    index= faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.float32(embeddings))
    faiss.write_index(index, f'indices/{lang}_{sent_id}')


def cosine_similarity(lang1_encodings, lang2_encodings):
    for i in range(lang1_encodings.shape[0]):
        lang1_vec = torch.tensor(lang1_encodings[i].squeeze())
        lang2_vec = torch.tensor(lang2_encodings[i].squeeze())    
        similarity = F.cosine_similarity(lang1_vec, lang2_vec, dim=0)
        print(similarity)
     
def closestPairs(embeds1, embeds2):
    data = embeds1.astype('float32')
    query_vector = embeds2.astype('float32')

    data = data / np.linalg.norm(data, axis=1, keepdims=True) # normalize
    query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(data.shape[1])
    index.add(data)
    distances, indices = index.search(query_vector, 1)

    # because we are just getting the nearest 1, don't need 2d list
    distances = [a[0] for a in distances]
    indices = [a[0] for a in indices]

    return distances, indices
        
def token_pair_similarity(data, lang1, lang2, sent_id, verbose=False, summary=False):

    # exclude language tag and <\s> token -- not sure whether I should 
    # these two consistently have by far the lowest max cosine similarity 
    # maybe worth checking langauge tag similarity for all language pairs?
    l1_data = (data[(lang1, sent_id)][0][1:-1], data[(lang1, sent_id)][1][1:-1])
    l2_data = (data[(lang2, sent_id)][0][1:-1], data[(lang2, sent_id)][1][1:-1])

    # token embeddings
    query_vector = l1_data[0]
    data = l2_data[0]

    print(query_vector.shape)
    exit()
    
    distancesAB, indices = closestPairs(data, query_vector)

    if verbose:
        print(f"{lang1} with {lang2}")
        for i in range(len(distancesAB)):
            print(f'{i}. {l1_data[1][i]}\t---\t{l2_data[1][indices[i]]}\t-\tsimilarity {distancesAB[i]:.3f}')
        
        print()
    
    query_vector = l2_data[0]
    data = l1_data[0]

    distancesBA, indices = closestPairs(data, query_vector)

    if verbose:
        print(f"{lang2} with {lang1}")
        for i in range(len(distancesBA)):
            print(f'{i}. {l2_data[1][i]} --- {l1_data[1][indices[i]]} - similarity {distancesBA[i]:.3f}')

    averaged_l1 = np.sum(np.array(l1_data[0]), axis=0) / len(l1_data[0]) # dividing not even needed for cosine
    averaged_l2 = np.sum(np.array(l2_data[0]), axis=0) / len(l2_data[0])

    angle = 1 - cosine(averaged_l1, averaged_l2)

    # could act as a kind of score for a language pair
    combined_mean = np.mean(distancesAB + distancesBA)

    if summary:
        print(f'average max angle between tokens A-B: {np.mean(distancesAB):.3f}')
        print(f'average max angle between tokens B-A: {np.mean(distancesBA):.3f}')
        print(f'average max angle between tokens A-B + B-A: {combined_mean:.3f}')
        print(f'angle between averaged embedding: {angle:.3f}')

    return combined_mean

def main():
    
    df = pd.read_pickle(SEED_EMBED_PICKLE) # had to do a pickle file :^|
    
    languages = TEN_SEED_LANGS
    # languages = NLLB_SEED_LANGS

    score_table = np.zeros((len(languages), len(languages)))

    data = {}
    # make dict for speed & cleanliness
    for index, row in df.iterrows():
        language = f"{row['language']}_{row['script']}"
        id = row['sent_id']
        data[(language, id)] = (row['embedding'], row['tokens'])

    same_script = []
    diff_script = []
    scores = []
    start = time.perf_counter()
    # your code here

    for i in range(0, len(languages)):
        for id in range(0, 20):
            make_index(data, languages[i], id)
        # for j in range(i + 1, len(languages)):

        #     lp_scores = []
        #     lang1=languages[i]
        #     lang2=languages[j]

        #     lang1_script = lang1.split('_')[1]
        #     lang2_script = lang2.split('_')[1]

        #     print(f'\n{lang1}, {lang2}')
        #     for id in range(0, 20):
                
            #     score = token_pair_similarity(data, lang1, lang2, id, verbose=False, summary=False)

            #     lp_scores.append(score)
            
            # score_table[i][j] = np.mean(lp_scores)
            # score_table[j][i] = np.mean(lp_scores)
            
            # print(f'average score between {lang1} and {lang2} is {np.mean(lp_scores):.3f}')

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")

    # make_heatmap(score_table, "test", '../plots/heatmaps', languages)

    # same_script_avg = np.mean(same_script)
    # diff_script_avg = np.mean(diff_script)

    print()
    # print(f'mean score for languages with same script: {same_script_avg:.3f}')
    # print(f'mean score for languages with different script: {diff_script_avg:.3f}')


if __name__ == "__main__":
    main()


"""
eng_Latn with fra_Latn
0. eng_Latn ---  une    - similarity 0.096
1. She      ---  Elle   - similarity 0.190
2. has      ---  une    - similarity 0.217
3. a        ---  une    - similarity 0.303
4. red      ---  rouge  - similarity 0.406
5. car      ---  voiture - similarity 0.437
6. .        ---  .      - similarity 0.226
7. </s>     ---  </s>   - similarity 0.033

fra_Latn with eng_Latn
0. fra_Latn --- She     - similarity 0.099
1. Elle     ---  She    - similarity 0.189
2. a        ---  has    - similarity 0.210
3. une      ---  a      - similarity 0.282
4. voiture  ---  car    - similarity 0.433
5. rouge    ---  red    - similarity 0.398
6. .        ---  .      - similarity 0.221
7. </s>     ---  </s>   - similarity 0.032

some results which make sense - 
looks like english, french, and german have much higher average maxes between
tokens than any of them with chinese. Whether this is because of script or other 
features, don't know. Can check with seed

But we see that the averaging of the token embeddings seems to erase a good amount of 
information in terms of angle. 

eng_Latn, fra_Latn
sentence 0
average angle between tokens: 0.350
angle between averaged embedding: 0.847

eng_Latn, deu_Latn
sentence 0
average angle between tokens: 0.362
angle between averaged embedding: 0.840

eng_Latn, zho_Hans
sentence 0
average angle between tokens: 0.266
angle between averaged embedding: 0.826

fra_Latn, deu_Latn
sentence 0
average angle between tokens: 0.361
angle between averaged embedding: 0.832

fra_Latn, zho_Hans
sentence 0
average angle between tokens: 0.252
angle between averaged embedding: 0.797

deu_Latn, zho_Hans
sentence 0
average angle between tokens: 0.251
angle between averaged embedding: 0.819



Ran experiment on all seed languages with sentences 100-199:

mean score for languages with same script: 0.099
mean score for languages with different script: 0.087

is that anything?
"""