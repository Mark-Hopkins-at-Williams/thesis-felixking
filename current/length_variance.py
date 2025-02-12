import os
import sys
import json
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats             # type: ignore
import matplotlib.pyplot as plt     # type: ignore

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

def make_diff_meaning_set(languages, input_dir, output_dir, num_sents, mean_stdev_dict):

    print('generating random means...')
    means = np.random.choice(list(mean_stdev_dict.keys()), size=num_sents, replace=True)

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

            count = 0
            accepted = False
            while not accepted:
                candidate = sents[random.randint(0, len(sents) - 1)]
                p = 2 * (1 - stats.norm.cdf(abs(mean - len(candidate)), 0, stdev))
                r = random.random()
                # print(abs(mean - len(candidate)), r, p)
                count += 1
                if count > 50000:
                    print("help")
                if r < p:
                    accepted = True
                    choices.append(candidate)

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

def token_num_variance(data, range_start, range_end, languages):
    vars = []
    for id in range(range_start, range_end):
        lens = [len(data[(languages[i], id)]) for i in range(0, len(languages))]
        vars.append(np.var(lens, ddof=1))

    avg_stdev = np.mean([np.sqrt(var) for var in vars])
    avg_var = np.mean(vars)
    print(f'average variance: {avg_var:.3f}')
    print(f'average standev:  {avg_stdev:.3f}')

def intralingual_length_var(data, range_start, range_end, languages):
    for lang in languages:
        lens = [len(data[(lang, id)]) for id in range(range_start, range_end)]
        variance = np.var(lens, ddof=1)
        print(lang)
        # check_normality(lens, lang, plot=True)
        print(f'\tvariance: {variance:.3f}')
        print(f'\tstandev:  {np.sqrt(variance):.3f}')

def generate_sent_group(len_stdev):
    mean = random.sample(list(len_stdev.keys()), 1)[0] # get a random mean from the list of means
    stdev = len_stdev[mean]
    rand_range = (int(-3*stdev), int(3*stdev))

    print(mean, stdev)

    for i in range(0, 100):
        test_len = random.randint(rand_range[0], rand_range[1])
        
        p = 2 * (1 - stats.norm.cdf(abs(test_len), 0, stdev))
        r = random.random()
        
        print(f'len: {test_len}, probability: {p:.2f}, \taccepted? {r < p}')



def main():

    config_file = sys.argv[1] 
    with open(config_file) as reader:
        config = json.load(reader)

    exp_dir = config['experiment_directory']
    
    range_start, range_end = config['sentence_range']
    languages = config['languages']
    df = pd.read_csv(config['parallel_corpus_csv'])
    mean_stdev_dict = string_len_variance(df, range_start, range_end)
    source_dir = config['line_by_line_dir']
    save_dir = f'{"/".join(source_dir.split("/")[:-1])}/scrambled_{source_dir.split("/")[-1]}'
    print(save_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    num_sents = range_end - range_start + 1

    mean_stdev = np.mean(list(mean_stdev_dict.values()))
    print(mean_stdev)

    if len(sys.argv) > 2 and sys.argv[2] == 'check':
        exit()

    make_diff_meaning_set(languages, source_dir, save_dir, num_sents, mean_stdev_dict)

if __name__ == "__main__":
    main()


"""
for each dataset, determine the standard deviation and the range of mean lens
then to generate a set of random different-meaning sentences with lenths matching this distribution, 
randomly choose a mean in the range, go through random sentences for each language, and based on the
randomly chosen sentence's length, use normal distribution to determine whether to add it


df = pd.read_pickle(config['parallel_corpus_pkl'])
df = df[(df.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (range_start <= df['sent_id']) & (df['sent_id'] <= range_end)]
data = {}
for _, row in df.iterrows():
    language = f"{row['language']}_{row['script']}"
    id = row['sent_id']
    data[(language, id)] = row['600M_embedding']

if parallel:
    print('computing length variances of parallel sentences...')
    token_num_variance(data, range_start, range_end, languages)

if by_lang:
    print('\ncomputing length variances of sentences in each language...')
    intralingual_length_var(data, range_start, range_end, languages)
"""