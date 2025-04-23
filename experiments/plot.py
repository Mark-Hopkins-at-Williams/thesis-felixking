from scipy.stats import pearsonr# type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns           # type: ignore
import pandas as pd
import numpy as np
import sys
import os

def plot(x, y, title, xlabel, ylabel, filename):
    correlation, _ = pearsonr(x, y)
    data = pd.DataFrame({'x': x, 'y': y})

    plt.figure(figsize=(8, 6))
    sns.regplot(x='x', y='y', data=data, ci=95, line_kws={"color": "red"})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.text(
        0.05, 0.95,  # x and y coordinates in axes fraction
        f"Correlation: {correlation:.2f}",
        fontsize=12,
        ha='left',
        va='center',
        transform=plt.gca().transAxes,  # Place text relative to axes
    )

    slope, intercept = np.polyfit(x, y, 1)

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.close()
    

def plot_slate(language, dir):
    
    scores = pd.read_csv(os.path.join(exp_dir, f'all-{language["tag"]}_scores.csv'))
    similarities = pd.read_csv(os.path.join(exp_dir, 'similarities.csv'))

    mask = (similarities['language'] != language['tag']) & (similarities['language'] != 'eng_Latn')
    scores = scores[mask]
    similarities = similarities[mask]   
    bleu = scores['bleu']
    chrf = scores['chrf++']
    avg_sim = similarities['avg_sim']
    lang_sim = similarities[f'{language["short"]}_sim']

    plot_dir = os.path.join(dir, language['name'])
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plot(bleu, avg_sim, 
        'Bleu Score vs Average Similarity Score', 
        'Bleu Scores', 
        'Encoding Similarity', 
        os.path.join(plot_dir, f'bleu_vs_avg_sim.png'))
    
    plot(chrf, avg_sim, 
        'chrf++ Score vs Average Similarity Score', 
        'chrf++ Scores', 
        'Encoding Similarity', 
        os.path.join(plot_dir, f'chrF_vs_avg_sim.png'))

    plot(bleu, lang_sim, 
        f'Bleu Score vs {language["name"]} Similarity Score', 
        'Bleu Scores', 
        f'{language["name"]} Similarity', 
        os.path.join(plot_dir, f'bleu_vs_{language["short"]}_sim.png'))
    
    plot(chrf, lang_sim, 
        f'chrf++ Score vs {language["name"]} Similarity Score', 
        'chrf++ Scores', 
        f'{language["name"]} Similarity', 
        os.path.join(plot_dir, f'chrF_vs_{language["short"]}_sim.png'))



if __name__ == '__main__':
    
    exp_dir = sys.argv[1]
    english = {'tag': 'eng_Latn', 'name': 'English', 'short': 'eng'}
    magahi = {'tag': 'mag_Deva', 'name': 'Magahi', 'short': 'mag'}
    czech = {'tag': 'ces_Latn', 'name': 'Czech', 'short': 'ces'}

    plot_slate(english, exp_dir)
    # plot_slate(czech, exp_dir)




