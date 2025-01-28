import os
import random
import torch    # type: ignore
import numpy as np
import pandas as pd
from tqdm import tqdm
from bottle import CustomM2M100Model
from finetuning.finetune import tokenize    # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

EP_PATH = '/mnt/storage/fking/data/europarl'
EP_FILTERED_PATH = '/mnt/storage/fking/data/europarl/filtered'
EP_UNFILTERED_PATH = '/mnt/storage/fking/data/europarl/unfiltered'

ep_to_nllb = {
    'bg': 'bul_Cyrl',
    'cs': 'ces_Latn',
    'da': 'dan_Latn',
    'de': 'deu_Latn',
    'el': 'ell_Grek',
    'en': 'eng_Latn',
    'es': 'spa_Latn',
    'et': 'est_Latn',
    'fi': 'fin_Latn',
    'fr': 'fra_Latn',
    'hu': 'hun_Latn',
    'it': 'ita_Latn',
    'lt': 'lit_Latn',
    'lv': 'lvs_Latn',
    'nl': 'nld_Latn',
    'pl': 'pol_Latn',
    'pt': 'por_Latn',
    'ro': 'ron_Latn',
    'sk': 'slk_Latn',
    'sl': 'slv_Latn',
    'sv': 'swe_Latn'
}

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

def add_embeddings(langs, df, size, token_text=False):    
    base_model = "facebook/nllb-200-distilled-600M"

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

    # df = coalesce_df(df, problems, size)
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


def make_df(ep_code, subset, train):

    lang, script = ep_to_nllb[ep_code].split('_')

    all_sentences = []
    with open(os.path.join(EP_FILTERED_PATH, f'{ep_code}_sents.txt'), 'r') as file:
        for line in file:
            all_sentences.append(line.strip())

    sents = [all_sentences[i] for i in subset]

    df = pd.DataFrame(columns=['language', 'script', 'sent_id', 'text'])
    df['sent_id'] = [x for x in range(0, len(subset))]
    df['script']  = script
    df['language'] = lang
    df['text'] = sents
    if train:
        df['split'] = 'train'
    return df
    
def compile(num_sents, name, num_sets, make_csv=False, train=False):
    all_langs = list(ep_to_nllb.values())
    for i in range(1, num_sets + 1):

        subset = random.sample(range(0, 183989), num_sents)
        frames = []

        for file in os.listdir(EP_FILTERED_PATH):
            if file[-3:] == 'txt':
                ep_code = file[0:2]
                frames.append(make_df(ep_code, subset, train))

        combined = pd.concat(frames, ignore_index=True)
        tag = ''
        if num_sets != 1:
            tag = f'-{i}'
        if make_csv:
            combined.to_csv(os.path.join(EP_PATH, f'{name}{tag}.csv'), index=False)
        combined = add_embeddings(all_langs, combined, num_sents)
        combined.to_pickle(os.path.join(EP_PATH, f'{name}{tag}.pkl'))
    
if __name__ == '__main__':

    compile(10000, 'europarl_10k', 1, make_csv=True, train=True)
