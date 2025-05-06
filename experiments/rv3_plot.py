from scipy.stats import pearsonr# type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns           # type: ignore
import pandas as pd
import numpy as np
import sys
import os
import re

def get_intra_sims(path):
    
    text = ''
    with open(os.path.join(path, '600M/intralingual_similarities.txt')) as file:
        text = file.read().strip()

    pattern = r'\n?(\D+): (\d+\.\d+)'

    matches = re.finditer(pattern, text)
    pairs = {}
    for match in matches:
        pairs[match.group(1)] = float(match.group(2))

    return pairs

def get_sims(path):

    df = pd.read_csv(os.path.join(path, '600M/similarities.csv'))
    pairs = dict(zip(df['language'], df['avg_sim']))

    return pairs

def plot(keys, vals1, vals2, dir):

    bar_width = 0.35
    r1 = np.arange(len(keys))
    r2 = [x + bar_width for x in r1]

    plt.figure(figsize=(15, 8))
    bars1 = plt.bar(r1, vals1, width=bar_width, label='intralingual')
    bars2 = plt.bar(r2, vals2, width=bar_width, label='spurious')

    plt.xlabel('Languages')
    plt.ylabel('Encoding Similarity')
    plt.title('Intralingual vs Spurious Similarity')
    plt.xticks([r + bar_width/2 for r in range(len(keys))], keys, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(dir, 'rv3.png'), dpi = 300)
    plt.close()

if __name__ == '__main__':

    exp_name = ''
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    else:
        print('usage: python rv1_plot.py <exp_name>')
        exit()

    spurious = f'scrambled_{exp_name}'

    if not os.path.isdir(spurious) or not os.path.isdir(exp_name):
        print('path doesn\'t exist')
        exit()

    intra = get_intra_sims(exp_name)
    spur = get_sims(spurious)
    keys = sorted(list(spur.keys()))

    vals1 = [intra[k] for k in keys]
    vals2 = [spur[k] for k in keys]

    print(f'intralingual mean: {np.mean(vals1):.3f}')
    print(f'spurious mean:     {np.mean(vals2):.3f}')

    print(f'intralingual variance: {np.var(vals1):.3E}')
    print(f'spurious variance:     {np.var(vals2):.3E}')


    plot(keys, vals1, vals2, '.')