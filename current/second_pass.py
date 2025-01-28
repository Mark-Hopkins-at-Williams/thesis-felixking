import re
from tqdm import tqdm
import torch    # type: ignore
from os import listdir
from os.path import join
import torch.nn.functional as F             # type: ignore
from bottle import CustomM2M100Model
from finetuning.finetune import tokenize    # type: ignore
from contrast import get_sentence_embeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
PARALLEL_TEXT_DIR = '/mnt/storage/fking/data/europarl/filtered'
MAX_TOKENS = 128

def remove_long_sents(corpora):
    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    too_long = set()
    for lang in tqdm(list(corpora.keys())):
        sents = corpora[lang]
        for i, sent in enumerate(sents):
            tokens = tokenize(sent, ep_to_nllb[lang], tokenizer, max_length=128)
            if len(tokens.input_ids[0]) >= MAX_TOKENS:
                too_long.add(i)
    
    for key in corpora:
        corpora[key] = [x for i, x in enumerate(corpora[key]) if i not in too_long]

    with open('toolong.txt', 'w') as file:
        for n in too_long:
            file.write(f'{n}\n')

    print(f'{len(too_long)}/{len(sents)} sentences are too long')


def checkForTokenizationIssues(corpora, batch_size=16):
    base_model = "facebook/nllb-200-distilled-600M"
    print("Loading NLLB model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    replacement_model = CustomM2M100Model(model.model.config)
    replacement_model.load_state_dict(model.model.state_dict())
    model.model = replacement_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval() 

    problems = set()
    for lang in tqdm(list(ep_to_nllb.keys())):
        embeddings = []
        code, _ = ep_to_nllb[lang].split('_')
        sents = corpora[lang]

        batchno = 1
        for i in range(0, len(sents), batch_size):
            batch = sents[i:min(i+batch_size, len(sents))]
            try:
                next_embeddings = get_sentence_embeddings(model, tokenizer, batch, code)
                # print(f'got batch {batchno} => {batchno*batch_size} sents')
            except Exception:
                next_embeddings = []
                for index, sent in enumerate(batch):
                    try:
                        next_embeddings.append(get_sentence_embeddings(model, tokenizer, sent, code))
                    except Exception:
                        print((batchno-1)*batch_size + index)
                        problems.add((batchno-1)*batch_size + index)
                        next_embeddings.append([0])          
                
            batchno += 1 
                        
            embeddings.extend(next_embeddings)

        for index, embedding in enumerate(embeddings):
            if len(embedding) >= MAX_TOKENS:
                problems.add(index)

    print('token problems:', problems)
    for key in corpora:
        corpora[key] = [x for i, x in enumerate(corpora[key]) if i not in problems]

def main():

    sentences = {key: [] for key in ep_to_nllb}

    for corpus in listdir(PARALLEL_TEXT_DIR):
        with open(join(PARALLEL_TEXT_DIR, corpus)) as file:
            for line in file:
                sentences[corpus[0:2]].append(line.strip())

    print(len(sentences['bg']))
    remove_long_sents(sentences)

    for tag in ep_to_nllb:
        with open(join(PARALLEL_TEXT_DIR, f'{tag}_sents.txt'), 'w') as file:
            for sent in sentences[tag]:
                file.write(sent + '\n')

if __name__ == '__main__':
    main()