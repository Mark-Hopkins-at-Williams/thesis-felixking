import re
import time
import random
from tqdm import tqdm
import torch    # type: ignore
from os import listdir
from os.path import join
import torch.nn.functional as F             # type: ignore
from bottle import CustomM2M100Model
from finetuning.finetune import tokenize    # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ep_to_nllb = {
    'bg': 'bul_Cyrl', 'cs': 'ces_Latn', 'da': 'dan_Latn', 'de': 'deu_Latn',
    'el': 'ell_Grek', 'en': 'eng_Latn', 'es': 'spa_Latn', 'et': 'est_Latn',
    'fi': 'fin_Latn', 'fr': 'fra_Latn', 'hu': 'hun_Latn', 'it': 'ita_Latn',
    'lt': 'lit_Latn', 'lv': 'lvs_Latn', 'nl': 'nld_Latn', 'pl': 'pol_Latn',
    'pt': 'por_Latn', 'ro': 'ron_Latn', 'sk': 'slk_Latn', 'sl': 'slv_Latn',
    'sv': 'swe_Latn'
}
europarl_languages = [
    'bul_Cyrl', 'ces_Latn', 'dan_Latn', 'deu_Latn',
    'ell_Grek', 'eng_Latn', 'spa_Latn', 'est_Latn',
    'fin_Latn', 'fra_Latn', 'hun_Latn', 'ita_Latn',
    'lit_Latn', 'lvs_Latn', 'nld_Latn', 'pol_Latn',
    'por_Latn', 'ron_Latn', 'slk_Latn', 'slv_Latn',
    'swe_Latn'
]

seed_languages = [
    "pbt_Arab", "bho_Deva", "nus_Latn", "ban_Latn", "dzo_Tibt", "mni_Beng", "lim_Latn", 
    "ltg_Latn", "ace_Latn", "crh_Latn", "srd_Latn", "taq_Latn", "mri_Latn", "ary_Arab", 
    "bam_Latn", "knc_Arab", "eng_Latn", "knc_Latn", "dik_Latn", "prs_Arab", "bjn_Arab", 
    "vec_Latn", "fur_Latn", "kas_Deva", "kas_Arab", "arz_Arab", "lij_Latn", "ace_Arab", 
    "bjn_Latn", "scn_Latn", "bug_Latn", "lmo_Latn", "szl_Latn", "hne_Deva", "fuv_Latn", 
    "taq_Tfng", "shn_Mymr", "mag_Deva"]

EUROPARL_TEXT_DIR = '/mnt/storage/fking/data/europarl_line_by_line'
EUROPARL_SCRAMBLED_DIR = '/mnt/storage/fking/data/scrambled_europarl_line_by_line'
SEED_TEXT_DIR = '/mnt/storage/fking/data/seed_line_by_line'
SEED_SCRAMBLED_DIR = '/mnt/storage/fking/data/scrambled_seed_line_by_line'
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

def main():

    sentences = {key: [] for key in ep_to_nllb}

    for corpus in listdir(EUROPARL_TEXT_DIR):
        with open(join(EUROPARL_TEXT_DIR, corpus)) as file:
            for line in file:
                sentences[corpus[0:2]].append(line.strip())

    print(len(sentences['bg']))
    remove_long_sents(sentences)

    for tag in ep_to_nllb:
        with open(join(EUROPARL_TEXT_DIR, f'{tag}_sents.txt'), 'w') as file:
            for sent in sentences[tag]:
                file.write(sent + '\n')

if __name__ == '__main__':
    main()