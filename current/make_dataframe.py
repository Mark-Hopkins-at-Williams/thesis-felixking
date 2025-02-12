import os
import random
import sys
import torch    # type: ignore
import numpy as np
import pandas as pd
from tqdm import tqdm
from bottle import CustomM2M100Model
from finetuning.finetune import tokenize    # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def add_embeddings(langs, df, size='600M'):    
    base_model = f"facebook/nllb-200-distilled-{size}"

    print("Loading NLLB model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, clean_up_tokenization_spaces=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    replacement_model = CustomM2M100Model(model.model.config)
    replacement_model.load_state_dict(model.model.state_dict())
    model.model = replacement_model

    embeddings = analyze_sentences(model, tokenizer, df, langs)
    embeddings_col = []
    
    print('getting embeddings...')
    for index, row in tqdm(df.iterrows()):
        embeddings_col.append(embeddings[f"{row['language']}_{row['script']}"][row['sent_id']])
        sentence = row['text']

    df[f'{size}_embedding'] = embeddings_col
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

    for lang in tqdm(langs):
        embeddings = []
        code, script = lang.split('_')
        sents = sentence_data[(sentence_data['language'] == code) & (sentence_data['script'] == script)]

        batchno = 1
        for i in range(0, len(sents), batch_size):
            batch = sents.iloc[i:i+batch_size]['text'].to_list()
            next_embeddings = get_sentence_embeddings(model, tokenizer, batch, code)        
                
            batchno += 1 
                        
            embeddings.extend(next_embeddings)

        all_embeddings[lang] = embeddings
    return all_embeddings


def make_df(corpus_dir, lang_code, subset, train):

    lang, script = lang_code.split('_')

    all_sentences = []
    with open(os.path.join(corpus_dir, f'{lang_code}_sents.txt'), 'r') as file:
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
    
def compile(num_sents, corpus_dir, output_stem, num_sets, word_count, make_csv=False, train=False):
    
    for i in range(1, num_sets + 1):

        subset = random.sample(range(0, word_count), num_sents)
        frames = []

        all_langs = []
        for file in os.listdir(corpus_dir):
            if file[-3:] == 'txt':
                lang_code = file[0:8]
                all_langs.append(lang_code)
                frames.append(make_df(corpus_dir, lang_code, subset, train))

        combined = pd.concat(frames, ignore_index=True)
        tag = ''
        if num_sets != 1:
            tag = f'-{i}'
        if make_csv:
            combined.to_csv(f'{output_stem}{tag}.csv', index=False)
        combined = add_embeddings(all_langs, combined, size='600M')
        combined = add_embeddings(all_langs, combined, size='1.3B')
        combined.to_pickle(f'{output_stem}{tag}.pkl')
    

def main():
    num_sents = 10000
    if len(sys.argv) > 1:
        num_sents = int(sys.argv[1])
    corpus_dir = '/mnt/storage/fking/data/europarl_line_by_line'
    # corpus_dir = '/mnt/storage/fking/data/seed_line_by_line'
    if len(sys.argv) > 2:
        corpus_dir = sys.argv[2]
    files = [os.path.join(corpus_dir, file) for file in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, file))]
    word_count = 0 
    with open(files[0]) as reader:
        for line in reader:
            word_count += 1
    print(f'wc: {word_count}')
    output_stem = './europarl_10k'
    if len(sys.argv) > 3:
        output_stem = sys.argv[3]
    print(f'Corpus dir: {corpus_dir}')
    print(f'Output stem: {output_stem}')
    compile(num_sents, corpus_dir, output_stem, 1, word_count, make_csv=True, train=True)
    

europarl_languages = [
    'bul_Cyrl', 'ces_Latn', 'dan_Latn', 'deu_Latn',
    'ell_Grek', 'eng_Latn', 'spa_Latn', 'est_Latn',
    'fin_Latn', 'fra_Latn', 'hun_Latn', 'ita_Latn',
    'lit_Latn', 'lvs_Latn', 'nld_Latn', 'pol_Latn',
    'por_Latn', 'ron_Latn', 'slk_Latn', 'slv_Latn',
    'swe_Latn'
]

if __name__ == '__main__':

    main()
    

# python make_dataframe.py 5 /mnt/storage/fking/data/europarl_line_by_line ./test_data/europarl_small
# python make_dataframe.py 5 /mnt/storage/fking/data/seed_line_by_line ./test_data/seed_small

