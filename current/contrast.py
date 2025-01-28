import sys

import torch    # type: ignore
import faiss    # type: ignore
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from finetuning.finetune import tokenize    # type: ignore
import torch.nn.functional as F             # type: ignore
from heatmaps import make_heatmap
from bottle import CustomM2M100Model
from scipy.spatial.distance import cosine   # type: ignore
from configure import NLLB_SEED_CSV, NLLB_SEED_LANGS, SEED_EMBED_PICKLE, TEN_SEED_LANGS 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

def unwrap_if_needed(embedding):
    if len(embedding) == 1:  # If it's a length-1 list
        return embedding[0]  # Return the inner list
    return embedding        # Otherwise return as-is

def coalesce_df(df, problems, size):


    total_size = len(df)
    print('removing problems...')
    for problem in problems:
        df = df[df['sent_id'] != problem]

    filtered_size = len(df)
    sents_per_lang = filtered_size // ((total_size-filtered_size) // len(problems))
    
    print('coalescing...')
    df['sent_id'] = np.arange(len(df)) % sents_per_lang
    
    if sents_per_lang < size:
        print("can't trim any sentences from df")
    else:
        df = df[(df['sent_id'] < size)]

    df = df.reset_index(drop=True) # convenient
    df['embedding'] = df['embedding'].apply(unwrap_if_needed) # necessary

    return df

def make_dataframe(langs, csv_path, size, token_text=False):    
    base_model = "facebook/nllb-200-distilled-600M"

    # the important parameters
    df = pd.read_csv(csv_path)
    # df = df[df['split'] == 'train']

    print("Loading NLLB model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    replacement_model = CustomM2M100Model(model.model.config)
    replacement_model.load_state_dict(model.model.state_dict())
    model.model = replacement_model

    embeddings, problems = analyze_sentences(model, tokenizer, df, langs)
    embeddings_col = []
    if token_text:
        tokens_col = []

    print('getting embeddings...')
    for index, row in tqdm(df.iterrows()):
        embeddings_col.append(embeddings[f"{row['language']}_{row['script']}"][row['sent_id']])

        sentence = row['text']
        if token_text:
            tokenized = tokenize(sentence, f"{row['language']}_{row['script']}", tokenizer, 128)
            tokens = [tokenizer.decode(token_id) for token_id in tokenized.input_ids[0]]
            tokens_col.append(tokens)

    df['embedding'] = embeddings_col
    if token_text:
        df['tokens'] = tokens_col

    df = coalesce_df(df, problems, size)
    print(len(df))

    return df


def get_sentence_embeddings(model, tokenizer, sents, language, max_length = 128):
    
    inputs = tokenize(sents, language, tokenizer, max_length=max_length).to(model.device)
    with torch.no_grad():
        model(**inputs, labels=inputs.input_ids)
        encoder_states = model.model.custom_module.snapshot
        attention_mask = inputs.attention_mask  
        result = []

        for i in range(encoder_states.shape[0]):
            num_valid_encoder_states = attention_mask[i].sum(dim=0).item()
            valid_vectors = encoder_states[i][:num_valid_encoder_states].cpu().numpy()
            result.append(valid_vectors)
    
    # result is list of token embeddings for a sentence
    return result


def analyze_sentences(model, tokenizer, sentence_data, langs, batch_size = 16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval() 
    all_embeddings = dict() 

    problems = set()
    for lang in tqdm(langs):
        embeddings = []
        code, script = lang.split('_')
        sents = sentence_data[(sentence_data['language'] == code) & (sentence_data['script'] == script)]

        batchno = 1
        for i in range(0, len(sents), batch_size):
            batch = sents.iloc[i:i+batch_size]['text'].to_list()
            try:
                next_embeddings = get_sentence_embeddings(model, tokenizer, batch, code)
                # print(f'got batch {batchno} => {batchno*batch_size} sents')
            except Exception as e:
                print(f'error on line {(batchno-1)*batch_size + index} in {lang}')
                next_embeddings = []
                for index, sent in enumerate(batch):
                    try:
                        next_embeddings.append(get_sentence_embeddings(model, tokenizer, sent, code))
                    except Exception:
                        problems.add((batchno-1)*batch_size + index)
                        next_embeddings.append([0])
                        
                
            batchno += 1 
                        
            embeddings.extend(next_embeddings)

        all_embeddings[lang] = embeddings
    return all_embeddings, problems


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
    
    # languages = TEN_SEED_LANGS
    languages = NLLB_SEED_LANGS
    sentence_range = range(0, 200)

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
        for j in range(i + 1, len(languages)):

            lp_scores = []
            lang1=languages[i]
            lang2=languages[j]

            lang1_script = lang1.split('_')[1]
            lang2_script = lang2.split('_')[1]

            # print(f'\n{lang1}, {lang2}')
            for id in sentence_range:

                score = token_pair_similarity(data, lang1, lang2, id, verbose=False, summary=False)

                lp_scores.append(score)
            
            score_table[i][j] = np.mean(lp_scores)
            score_table[j][i] = np.mean(lp_scores)
            
            print(f'average score between {lang1} and {lang2} is {np.mean(lp_scores):.3f}')

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")

    make_heatmap(score_table, "test", '../plots/heatmaps', languages)

    # same_script_avg = np.mean(same_script)
    # diff_script_avg = np.mean(diff_script)

    print()
    # print(f'mean score for languages with same script: {same_script_avg:.3f}')
    # print(f'mean score for languages with different script: {diff_script_avg:.3f}')

ep_langs = [
    'bul_Cyrl',
    'ces_Latn',
    'dan_Latn',
    'deu_Latn',
    'ell_Grek',
    'eng_Latn',
    'spa_Latn',
    'est_Latn',
    'fin_Latn',
    'fra_Latn',
    'hun_Latn',
    'ita_Latn',
    'lit_Latn',
    'lvs_Latn',
    'nld_Latn',
    'pol_Latn',
    'por_Latn',
    'ron_Latn',
    'slk_Latn',
    'slv_Latn',
    'swe_Latn'
]

if __name__ == "__main__":
    csv_path = sys.argv[1]
    pkl_path = csv_path[:-3] + "pkl"
 
    num_sents_to_keep = int(sys.argv[2])

    df = make_dataframe(ep_langs, csv_path, num_sents_to_keep)
    
    print('writing to file...')
    df.to_pickle(pkl_path)
