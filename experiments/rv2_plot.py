import os
import re
import sys
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

def get_pairwise_similarities(path):
    text = ''
    with open(os.path.join(path, 'data.txt')) as file:
        text = file.read()

    pattern = r'\n?(\D+), (\D+): (\d+\.\d+)'
    matches = re.finditer(pattern, text)

    languages = set()
    data = {}

    for match in matches:
        lang1 = match.group(1)
        lang2 = match.group(2)
        sim = match.group(3)
        
        languages.update([lang1,lang2])

        data[(lang1, lang2)] = float(sim)
        data[(lang2, lang1)] = float(sim)

    score_table = np.full((len(languages), len(languages)), 1.0)
    langs = list(languages)
    for i in range(0, len(langs)):
        for j in range (i+1, len(langs)):
            score_table[i][j] = data[(langs[i], langs[j])]
            score_table[j][i] = data[(langs[i], langs[j])]

    return score_table, langs


def make_heatmap(data, title, output_dir, labels):

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        cluster_grid = sns.clustermap(
            data,
            cmap="viridis",
            annot=False,
            xticklabels=labels,
            yticklabels=labels,
            figsize=(10, 8),
            dendrogram_ratio=(.1, .1)
        )

    row_order = cluster_grid.dendrogram_row.reordered_ind
    col_order = cluster_grid.dendrogram_col.reordered_ind

    ordered_data = data[np.ix_(row_order, col_order)]


    ordered_xticks = [labels[i] for i in row_order]
    ordered_yticks = [labels[i] for i in col_order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        ordered_data,
        annot=False,
        xticklabels=ordered_xticks,
        yticklabels=ordered_yticks,
        cmap="viridis",
        ax=ax)

    ax.set_xlabel("Languages")
    ax.set_ylabel("Languages")

    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.xticks(rotation=60, ha='left') 

    plt.savefig(
        os.path.join(output_dir, f'{title}_heatmap'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


if __name__ == '__main__':
    path = ''
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print('usage: python rv2_plot.py <exp_dir_path>')
        exit()

    if not os.path.isdir(sys.argv[1]):
        print('path doesn\'t exist')
        exit()

    score_table, languages = get_pairwise_similarities(path)
    make_heatmap(score_table, 'test_heatmap', '.', languages)
