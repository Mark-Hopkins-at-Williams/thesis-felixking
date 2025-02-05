
import os
import sys
import json
import shutil
import evaluate                                                     # type: ignore
import numpy as np
import pandas as pd
from tqdm import tqdm
from heatmaps import make_heatmap
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from configure import NLLB_SEED_LANGS, SEED_EMBED_PICKLE, TEN_SEED_LANGS

def evaluate_translations(candidate_translations, reference_translations):
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result  = bleu_calc.compute(predictions=candidate_translations, references=reference_translations)
    chrf_result = chrf_calc.compute(predictions=candidate_translations, references=reference_translations)
    bleu_score = round(bleu_result['score'], 3)
    chrf_score = round(chrf_result['score'], 3)
    print(f"bleu: {bleu_score}\nchrf: {chrf_score}")

    return bleu_score, chrf_score


def main():
    config_file = sys.argv[1] 
    with open(config_file) as reader:
        config = json.load(reader)

    exp_dir = config['experiment_directory']
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    save_dir = os.path.join(exp_dir, 'translation_similarity')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    
    df = pd.read_csv(config['parallel_corpus_csv']) 
    primary = config['primary']
    languages = config['languages']
    languages = [l for l in languages if l != primary]
    range_start, range_end = config['sentence_range']
    # range_start, range_end = [0, 4]
    df = df[(df.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= df['sent_id']) & (df['sent_id'] <= range_end)]


    # ASK MARK -- should we actually just run eng->eng? or assume it's perfect? I think it wouldn't be perfect
    score_table = np.full((len(languages), len(languages)), 100.0) # bleu and chrf++ have max of 100, right?
    # using a default score of 100 makes the chart much harder to read. how to correct?
    # maybe first, save the similarities, otherwise, set the diagonal to average of the row or something? just something not
    # outside of the range

    max_bleu = 0
    print('computing similarities...')
    with open(os.path.join(save_dir, 'similarities.txt'), 'w') as file:
        for i in tqdm(range(0, len(languages))):
            for j in range(i + 1, len(languages)):

                l1, s1 = languages[i].split('_')
                l2, s2 = languages[j].split('_')
                
                l1_translations = list(df[(df['language'] == l1) & (df['script'] == s1)]['600M_eng_Latn_translations'])
                l2_translations = list(df[(df['language'] == l2) & (df['script'] == s2)]['600M_eng_Latn_translations'])

                bleu, chrf = evaluate_translations(l1_translations, l2_translations)
                max_bleu = max(bleu, max_bleu)
                score_table[i][j] = bleu
                score_table[j][i] = bleu
                file.write(f'{languages[i]}, {languages[j]}: {bleu}')

    for i in range(0, len(languages)):
        score_table[i][i] = max_bleu    # compromise

    make_heatmap(score_table, "unordered", save_dir, languages)
    make_heatmap(score_table, "clustered", save_dir, languages, cluster=True)

    shutil.copy(config_file, os.path.join(exp_dir, os.path.basename(config_file)))

if __name__ == "__main__":
    main()