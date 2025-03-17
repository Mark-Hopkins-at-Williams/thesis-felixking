import os
import sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats             # type: ignore
from rapidfuzz import fuzz          # type: ignore
import matplotlib.pyplot as plt     # type: ignore
from make_dataframe import add_embeddings

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

SEED_PATH = '/mnt/storage/fking/data/seed'
EUROPARL_PATH = '/mnt/storage/fking/data/europarl'

def too_similar(sent1, sent2, threshold=70):
    return fuzz.ratio(sent1, sent2) > threshold

def sample_by_distribution(mean, stdev, sents):

    iters = 0
    while True:
        candIndex = random.randint(0, len(sents) - 1)
        candidate = sents[candIndex]

        p = 2 * (1 - stats.norm.cdf(abs(mean - len(candidate)), 0, stdev))
        r = random.random()
        if r < p:   # accepted
            return candIndex

        if iters % 1000 == 0 and iters % 1000 != 0:
            print(iters)

        iters += 1
        if iters > 100000:
            print('tested 100,000 candidates with no successes. Exiting...')
            exit()
        
def RV2(languages, input_dir, output_path, num_sents, mean_stdev_dict):

    dfA = pd.DataFrame()
    dfB = pd.DataFrame()

    for language in tqdm(languages):
        sents = []
        with open(os.path.join(input_dir, f'{language}_sents.txt'), 'r') as file:
            for line in file:
                sents.append(line)

        As = random.sample(sents, num_sents)
        for i, s in enumerate(As): # kinda clunky
            while len(s) not in mean_stdev_dict:
                s = random.choice(sents)
            
            As[i] = s        

        Bs = []
        for sent in As:
            sent_len = len(sent)
            choice = sent
            while too_similar(choice, sent):
                choice = sents[sample_by_distribution(sent_len, mean_stdev_dict[sent_len], sents)]
            Bs.append(choice)
            
        l, s = language.split('_')
        new_dfA = pd.DataFrame([{'language': l, 'script': s, 'sent_id':i, 'text': a.strip(), 'set': 'a'} for i, a in enumerate(As)])
        new_dfB = pd.DataFrame([{'language': l, 'script': s, 'sent_id':i, 'text': b.strip(), 'set': 'b'} for i, b in enumerate(Bs)])
        dfA = pd.concat([dfA, new_dfA])
        dfB = pd.concat([dfB, new_dfB])

    dfA.to_csv(f'{output_path}_A.csv', index=False)
    add_embeddings(languages, dfA, size='600M')
    add_embeddings(languages, dfA, size='1.3B')
    dfA.to_pickle(f'{output_path}_A.pkl')
    
    dfB.to_csv(f'{output_path}_B.csv', index=False)
    add_embeddings(languages, dfB, size='600M')
    add_embeddings(languages, dfB, size='1.3B')
    dfB.to_pickle(f'{output_path}_B.pkl')
    

# different meaning, different language
def RV3(languages, input_dir, output_dir, num_sents, mean_stdev_dict):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('generating random means...')
    means = np.random.choice(list(mean_stdev_dict.keys()), size=num_sents, replace=True)
    indexChoices = {i: set() for i in range(0, num_sents)}

    for language in languages:
        print(f'getting sentences for {language}')
        sents = []
        with open(os.path.join(input_dir, f'{language}_sents.txt'), 'r') as file:
            for line in file:
                sents.append(line)

        choices = []
        for id in tqdm(range(0, num_sents)):

            mean = means[id]
            stdev = mean_stdev_dict[mean]

            indexChoice = sample_by_distribution(mean, stdev, sents)
            while indexChoice in indexChoices[id]:
                indexChoice = sample_by_distribution(mean, stdev, sents)

            choice = sents[indexChoice]
            choices.append(choice)
            indexChoices[id].add(indexChoice)

        with open(os.path.join(output_dir, f'{language}_sents.txt'), 'w') as file:
            file.write(''.join(choices))

def string_len_variance(df, range_start, range_end):

    print('getting means and stdevs...')
    len_stdev = {}
    for id in tqdm(range(range_start, range_end + 1)):
        sents = list(df[df['sent_id'] == id]['text'])
        lens = [len(s) for s in sents]

        rounded = int(np.mean(lens))
        if rounded not in len_stdev:
            len_stdev[rounded] = []
        len_stdev[rounded].append(np.sqrt(np.var(lens, ddof=1)))

    for key in len_stdev:
        len_stdev[key] = np.mean(len_stdev[key])
    return len_stdev

def main():

    config_file = sys.argv[1] 
    with open(config_file) as reader:
        config = json.load(reader)
    
    range_start, range_end = config['sentence_range']
    languages = config['languages']
    df = pd.read_csv(config['parallel_corpus_csv'])
    source_dir = config['line_by_line_dir']
    rv3_save_dir = f'{"/".join(source_dir.split("/")[:-1])}/scrambled_{source_dir.split("/")[-1]}'
    rv2_save_path = config['parallel_corpus_csv'].split('.')[0] + '_intralingual_TEST'

    num_sents = range_end - range_start + 1

    mean_stdev_dict = string_len_variance(df, range_start, range_end)
    mean_stdev = np.mean(list(mean_stdev_dict.values()))
    mean_mean = np.mean(list(mean_stdev_dict.keys()))
    print(mean_stdev, mean_mean)

    if len(sys.argv) > 2:
        if sys.argv[2] == 'rv2':
            RV2(languages, source_dir, rv2_save_path, num_sents, mean_stdev_dict)
        elif sys.argv[2] == 'rv3':
            RV3(languages, source_dir, rv3_save_dir, num_sents, mean_stdev_dict)

if __name__ == "__main__":
    main()


"""
for each dataset, determine the standard deviation and the range of mean lens
then to generate a set of random different-meaning sentences with lenths matching this distribution, 
randomly choose a mean in the range, go through random sentences for each language, and based on the
randomly chosen sentence's length, use normal distribution to determine whether to add it

"""
