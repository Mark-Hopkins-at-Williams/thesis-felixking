import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from configure import SEED_EMBED_PICKLE

def filter_df(parquet_file, languages, sent_id_range = None, sample_size = None):

    df = pd.read_parquet(parquet_file)
    
    #filter by language
    df = df[(df.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (df['split'] == 'train')]

    # filter by sentence ID range if specified
    if sent_id_range:
        start_id, end_id = sent_id_range
        df = df[df['sent_id'].between(start_id, end_id)]
    
    # sample sentences if specified
    if sample_size:
        df = df.groupby('language').apply(
            lambda x: x.sample(min(len(x), sample_size), random_state=42)
        ).reset_index(drop=True)

    return df
    
def get_angle_and_dist_tables(data):

    langs = sorted(data['language'].unique())
    sents = sorted(data['sent_id'].unique())
    
    n_langs = len(langs)
    dist_table = np.zeros((n_langs, n_langs))
    angle_table = np.zeros((n_langs, n_langs))
    
    # Create a dictionary to store embeddings for faster lookup
    # Structure: {(language, sent_id): embedding}
    embedding_dict = dict(
        zip(zip(data['language'], data['sent_id']), data['embedding'])
    )
    
    # Iterate over upper triangle only
    for i in tqdm(range(n_langs)):
        for j in range(i, n_langs):
            if i == j:
                angle_table[i][j] = 1
                continue
                
            lang1, lang2 = langs[i], langs[j]
            print(f'{lang1} x {lang2}')
            
            # Vectorized computation for all sentences at once
            embeddings1 = np.array([embedding_dict[(lang1, sent)] for sent in sents])
            embeddings2 = np.array([embedding_dict[(lang2, sent)] for sent in sents])
            
            # Calculate distances and angles in bulk
            distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
            angles = np.array([1 - cosine(emb1, emb2) 
                             for emb1, emb2 in zip(embeddings1, embeddings2)])
            
            # Calculate averages
            dist_avg = np.mean(distances)
            angle_avg = np.mean(angles)
            
            # Fill both triangles at once
            dist_table[i, j] = dist_table[j, i] = dist_avg
            angle_table[i, j] = angle_table[j, i] = angle_avg
            
            print('angle average: ', angle_avg)
            print('distance average: ', dist_avg)
    
    return angle_table, dist_table

def make_heatmap(data, title, output_dir, labels):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data,
        annot=False,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        ax=ax)

    ax.set_xlabel("Languages")
    ax.set_ylabel("Languages")

    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.xticks(rotation=60, ha='left') 

    ax.set_title(title)

    plt.savefig(
        os.path.join(output_dir, f'{title}_heatmap'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


if __name__ == '__main__':

    table = pq.read_table(SEED_EMBED_PARQUET)
    data = table.to_pandas()


    # data = pd.read_csv(SEED_EMBED_PARQUET, engine='pyarrow')
    # exit()
    output_dir = 'plots/heatmaps'

    data['language'] = data['language'] + '_' + data['script']
    data = data.drop('script', axis=1)
    
    languages = sorted(np.unique(data['language']))


    # data = data[data['sent_id'] < 10]

    # embeddings = data['embedding']
    # embed0 = embeddings.iloc[0]


    # df = data[(data['language'] == 'eng_Latn') & (data['sent_id'] == 0)]
    # print(len(df))
    # print(len(df['embedding']))
    # eng0 = df['embedding']
    # for i in eng0:
    #     print(i)


    angle_table, dist_table = get_angle_and_dist_tables(data)

    make_heatmap(angle_table, 'Cosine', output_dir, languages)
    make_heatmap(dist_table, 'Distance', output_dir, languages)