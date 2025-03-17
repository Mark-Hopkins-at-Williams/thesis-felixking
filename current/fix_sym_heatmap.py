from heatmaps import make_heatmap
import re
import os
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('missing command line argument: data path')
        exit()

    input_path = sys.argv[1]
    output_dir = os.path.dirname(input_path)

    pattern = r'(\w+), (\w+): (\d+\.\d+)'
    with open(input_path, 'r') as file:
        text = file.read()
        matches = re.findall(pattern, text)

    languages = set()
    languages.update([l[0] for l in matches])    
    languages.update([l[1] for l in matches])  
    languages = list(languages)  
    max_bleu = max([float(l[2]) for l in matches])

    score_dict = {(m[0], m[1]): float(m[2]) for m in matches}
    symmetrical_score_table = np.full((len(languages), len(languages)), max_bleu)

    for i in range(0, len(languages)):
        for j in range(i+1, len(languages)):
            l1 = languages[i]
            l2 = languages[j]
            avg_score = (score_dict[(l1, l2)] + score_dict[(l2, l1)]) / 2
            symmetrical_score_table[i][j] = avg_score
            symmetrical_score_table[j][i] = avg_score

    make_heatmap(symmetrical_score_table, 'symmetrical_unordered', output_dir, languages)
    make_heatmap(symmetrical_score_table, 'symmetrical_clustered', output_dir, languages, cluster=True)

# python fix_sym_heatmap.py /mnt/storage/fking/thesis-felixking/experiments/europarl/translation_similarity/similarities.txt