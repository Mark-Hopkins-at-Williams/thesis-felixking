
import os
import sys
import json
import faiss # type: ignore
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from validate import batched_translate
from heatmaps import make_heatmap
from finetuning.finetune import tokenize # type: ignore
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from configure import NLLB_SEED_LANGS, SEED_EMBED_PICKLE, TEN_SEED_LANGS
from finetuning.multilingualdata import MultilingualCorpus          # type: ignore

def get_translations(csv_path, nllb_size, languages, target, save_path):
    
    print('loading model...')
    base_model = "facebook/nllb-200-distilled-" + nllb_size
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model).cuda()

    pairs = [(l, target) for l in languages]
    corpus = MultilingualCorpus(csv_path)
    data = pd.read_csv(csv_path)
    data['translation'] = ''
    for (s,t) in pairs:

        dev_bitext = corpus.create_bitext(s, t, 'train')   
        src_texts = dev_bitext.lang1_sents
        lang, script = s.split('_')
        mask = ((data['language'] == lang) & (data['script'] == script) & (data['split'] == 'train'))
        translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=dev_bitext.lang1_code, tgt_lang=dev_bitext.lang2_code)
        data.loc[mask, 'translation'] = translations

    data.to_csv(save_path, index=False)
    return data

def main():
    config_file = sys.argv[1] 
    with open(config_file) as reader:
        config = json.load(reader)

    exp_dir = config['experiment_directory']
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    else:
        raise Exception(f'Experiment directory already exists: {exp_dir}')


    print('loading sentences...')
    df = pd.read_csv(config['parallel_corpus_file']) 
    
    languages = config['languages']
    range_start, range_end = config['sentence_range']

    df = df[(df.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= df['sent_id'] <= range_end)]

    # ASK MARK -- should we actually just run eng->eng? or assume it's perfect? I think it wouldn't be perfect
    score_table = np.full((len(languages), len(languages)), 100.0) # bleu and chrf++ have max of 100, right?

    print('translating sentences...')
    data = {}   # make dict for speed & cleanliness
    for _, row in df.iterrows(): 
        language = f"{row['language']}_{row['script']}"
        id = row['sent_id']

        #do the batched translate

        translation = ""
        data[(language, id)] = translation


    print('computing similarities...')
    with open(os.path.join(exp_dir, 'similarities.txt'), 'w') as file:
        for i in tqdm(range(0, len(languages))):
            for j in range(i + 1, len(languages)):
                lp_scores = []
                lang1=languages[i]
                lang2=languages[j]
                for id in range(range_start, range_end):   
                    score = 0 # replace this with bleu or chrf calculation call
                    lp_scores.append(score)
                file.write(f'\n{lang1}, {lang2}: {np.mean(lp_scores)}')
                score_table[i][j] = np.mean(lp_scores)
                score_table[j][i] = np.mean(lp_scores)


    make_heatmap(score_table, "unordered", exp_dir, languages)
    make_heatmap(score_table, "clustered", exp_dir, languages, cluster=True)

    shutil.copy(config_file, os.path.join(exp_dir, os.path.basename(config_file)))

if __name__ == "__main__":
    ep_langs = [
    'bul_Cyrl', 'ces_Latn', 'dan_Latn', 'deu_Latn', 'ell_Grek', 'eng_Latn',
    'spa_Latn', 'est_Latn', 'fin_Latn', 'fra_Latn', 'hun_Latn', 'ita_Latn',
    'lit_Latn', 'lvs_Latn', 'nld_Latn', 'pol_Latn', 'por_Latn', 'ron_Latn',
    'slk_Latn', 'slv_Latn', 'swe_Latn'
    ]
    seed_langs = [
    'pbt_Arab', 'bho_Deva', 'nus_Latn', 'ban_Latn', 'dzo_Tibt', 'mni_Beng', 'lim_Latn', 
    'ltg_Latn', 'ace_Latn', 'crh_Latn', 'srd_Latn', 'taq_Latn', 'mri_Latn', 'ary_Arab', 
    'bam_Latn', 'knc_Arab', 'eng_Latn', 'knc_Latn', 'dik_Latn', 'prs_Arab', 'bjn_Arab', 
    'vec_Latn', 'fur_Latn', 'kas_Deva', 'kas_Arab', 'arz_Arab', 'lij_Latn', 'ace_Arab', 
    'bjn_Latn', 'scn_Latn', 'bug_Latn', 'lmo_Latn', 'szl_Latn', 'hne_Deva', 'fuv_Latn', 
    'taq_Tfng', 'shn_Mymr', 'mag_Deva']

    get_translations('/mnt/storage/fking/data/europarl/europarl_10k.csv', '600M', ep_langs, 'eng_Latn', '/mnt/storage/fking/data/europarl/europarl_10k_translations.csv')
    # main()