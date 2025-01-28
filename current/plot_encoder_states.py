import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from finetuning.finetune import tokenize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from hierarchical import perform_hierarchical_clustering, plot_clustering_results
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from itertools import combinations
from matplotlib.lines import Line2D
from bottle import CustomM2M100Model
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
from scipy.spatial.distance import cosine
from configure import NLLB_SEED_CSV, NLLB_SEED_LANGS, TEN_SEED_LANGS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_and_filter_data(
    csv_path,
    languages,
    sent_id_range = None,
    sample_size = None
):
    """Loads in the NLLB-SEED csv (or any csv with that structure).
    
    Filters out any unspecified languages and sentences outside of the provided range.    
    
    """
    df = pd.read_csv(csv_path)
    df = df[(df.apply(lambda row: f"{row['language']}_{row['script']}" in languages, axis=1)) & (df['split'] == 'train')] # filter by lang
    if sent_id_range: # filter by sentence ID range if specified
        start_id, end_id = sent_id_range
        df = df[df['sent_id'].between(start_id, end_id)]
    if sample_size: # sample sentences if specified
        df = df.groupby('language').apply(
            lambda x: x.sample(min(len(x), sample_size), random_state=42)
        ).reset_index(drop=True)
    
    return df


def get_sentence_embeddings(
    model,
    tokenizer,
    sents,
    language,
    max_length = 128
):

    # tokenize the input
    inputs = tokenize(sents, language, tokenizer, max_length=max_length).to(model.device)


    # Get encoder states
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        encoder_states = model.model.custom_module.snapshot

        attention_mask = inputs.attention_mask  

        # Average the token embeddings, excluding padding tokens
        masked_states = encoder_states * attention_mask.unsqueeze(-1)  
        sum_states = masked_states.sum(dim=1)   
        sum_mask = attention_mask.sum(dim=1, keepdim=True) 
        sentence_embedding = (sum_states / sum_mask).cpu().numpy()

    return sentence_embedding

def analyze_sentences(
    model,
    tokenizer,
    sentence_data,
    langs,
    batch_size = 16
):
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_embeddings = None
    
    # Process sentences in batches
    for lang in tqdm(langs):
        code = lang.split('_')[0]
        script = lang.split('_')[1]
        sents = sentence_data[(sentence_data['language'] == code) & (sentence_data['script'] == script)]
        # print(len(sents))
        # batchno = 1
        for i in range(0, len(sents), batch_size):

            batch = sents.iloc[i:i+batch_size]['text'].to_list()
            
            embeddings = get_sentence_embeddings(
                model,
                tokenizer,
                batch,
                code,
                max_length= 2048 // batch_size
            )
            # print('did batch', batchno, '=', batchno*batch_size, 'sents')
            # batchno += 1
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = np.vstack([all_embeddings, embeddings])
            
    return all_embeddings

def get_reduction(embeddings, perplexity=30, n_iter=1000, random_state=42):
    n_samples = len(embeddings)

    if perplexity is None or perplexity >= n_samples:
        # Rule of thumb: perplexity should be roughly n_samples / 100,
        # but not less than 5 or more than 50
        suggested_perplexity = max(5, min(50, n_samples // 100))
        print(f"Adjusting perplexity to {suggested_perplexity} based on dataset size")
        perplexity = suggested_perplexity

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings


def visualize_embeddings(
    metadata,
    reduced_embeddings,
    output_dir,
    dpi = 300,
    figsize = (15, 10),
):

    os.makedirs(output_dir, exist_ok=True)

    # Create color palette for languages
    unique_languages = metadata['language'] + '_' + metadata['script']
    unique_languages = sorted(unique_languages.unique())

    color_palette = sns.color_palette('husl', n_colors=len(unique_languages))
    color_dict = dict(zip(unique_languages, color_palette))

    # Cycle through markers for languages
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '8', 'x', '+']  # Different marker shapes
    # markers = ['o', 's', '^', 'x']  # Different marker shapes
    marker_dict = dict(zip(unique_languages, [markers[i % len(markers)] for i in range(len(unique_languages))]))

    # Plot by language
    plt.figure(figsize=figsize)
    
    for lang in unique_languages:
        mask = (metadata['language'] == lang.split("_")[0]) & (metadata['script'] == lang.split("_")[1])
        plt.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            label=lang,
            c=[color_dict[lang]],
            marker=marker_dict[lang],
            alpha=0.7,
            s=100
        )
    
    plt.title(f'Sentence Embeddings by Language (tsne)')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(output_dir, f'by_lang.png'),
        dpi=dpi,
        bbox_inches='tight'
    )
    plt.close()


    sent_ids = metadata['sent_id'].unique()

    if len(sent_ids) <= 20:
    
        color_palette_ids = sns.color_palette('husl', n_colors=len(sent_ids))
        color_dict_ids = dict(zip(sent_ids, color_palette_ids))
        # Plot by sentence ID (to see parallel sentences)
        
        plt.figure(figsize=figsize)

        for id in sent_ids:
            mask = metadata['sent_id'] == id
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                label=id,
                color=color_dict_ids[id],
                marker=markers[int(id) % len(markers)],
                alpha=0.7,
                s=100
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:

        plt.figure(figsize=figsize)

        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=metadata['sent_id'],
            cmap='viridis',
            alpha=0.7,
            s=100
        )
    
        plt.colorbar(scatter, label='Sentence ID')

    plt.title(f'Sentence Embeddings by Sentence ID (tsne)')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(output_dir, f'by_id.png'),
        dpi=dpi,
        bbox_inches='tight'
    )
    plt.close()
    
def get_subset_mean(subset):
    length = subset.shape[0]
    angles = []
    for i in range(0, length):
        embedding1 = subset[i]
        for j in range(i + 1, length):
            embedding2 = subset[j]
            angle = 1 - cosine(embedding1, embedding2)
            angles.append(angle)
        
    return np.mean(angles)

def get_disjoint_mean(set1, tag1, set2, tag2, memo):
    if (tag1, tag2) in memo:
        return memo[(tag1, tag2)]
    if (tag2, tag1) in memo:
        return memo[(tag2, tag1)]
    
    len1 = set1.shape[0]
    len2 = set2.shape[0]
    angles = []
    for i in range(0, len1):
        embedding1 = set1[i]
        for j in range(0, len2):
            embedding2 = set2[j]
            angle = 1 - cosine(embedding1, embedding2)
            angles.append(angle)
        
    mean = np.mean(angles)
    memo[(tag1, tag2)] = mean
    return mean

def get_cosine_similarity(embeddings, data, langs, pr=False):

    # calculate average angle between embeddings by language
    result = {}

    memo = {}
    for lang1 in langs:
        other_means = []
        mask1 = (data['language'] == lang1.split("_")[0]) & (data['script'] == lang1.split("_")[1])
        self = embeddings[mask1]
        for lang2 in [l for l in langs if l != lang1]:
            mask2 = (data['language'] == lang2.split("_")[0]) & (data['script'] == lang2.split("_")[1])
            other = embeddings[mask2]
            other_means.append(get_disjoint_mean(self, lang1, other, lang2, memo))
        
        other_mean = np.mean(other_means)
        self_mean = get_subset_mean(self)

        result[lang1] = (self_mean, other_mean)

        if pr:
            print(f'{lang1} self:  {self_mean:.3f}')
            print(f'{lang1} other: {other_mean:.3f}')

    return result

def get_avg_dist_by_language_pair(embeddings, data, langs, pr=False):

    # calculate average angle between embeddings by language
    result = {}

    memo = {}
    for lang1 in langs:
        other_means = []
        mask1 = (data['language'] == lang1.split("_")[0]) & (data['script'] == lang1.split("_")[1])
        self = embeddings[mask1]
        for lang2 in [l for l in langs if l != lang1]:
            mask2 = (data['language'] == lang2.split("_")[0]) & (data['script'] == lang2.split("_")[1])
            other = embeddings[mask2]
            other_means.append(get_disjoint_mean(self, lang1, other, lang2, memo))
        
        other_mean = np.mean(other_means)
        self_mean = get_subset_mean(self)

        result[lang1] = (self_mean, other_mean)

        if pr:
            print(f'{lang1} self:  {self_mean:.3f}')
            print(f'{lang1} other: {other_mean:.3f}')

    return result

def plot_cosine_similarity(data, output_dir, dpi = 300, figsize = (15, 10)):
    
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=figsize)

    languages = list(data.keys())
    values1 = [v[0] for v in data.values()]
    values2 = [v[1] for v in data.values()]

    # Set up the bar positions
    x = np.arange(len(languages))
    width = 0.35  # Width of the bars

    # Create the bars
    plt.bar(x - width/2, values1, width, label='Average self-similarity')
    plt.bar(x + width/2, values2, width, label='Average other-similarity')

    # Customize the plot
    plt.xlabel('Languages')
    plt.ylabel('Values')
    plt.title('Language Values Comparison')
    plt.xticks(x, languages, rotation=45)  # Rotate labels for better r
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f'cosine_similarity.png'),
        dpi=dpi,
        bbox_inches='tight'
    )
    plt.close()

def get_pairwise_dist(data, embeddings, languages, output_dir):
    
    for lang in languages:
        subset = embeddings[(data['language'] == lang.split('_')[0]) & (data['script'] == lang.split('_')[1])]

        distances = []
        min = 1000000
        min_pair = None
        max = 0
        max_pair = None
        for i in range(0, len(subset)):
            vec1 = subset[i]
            for j in range(i+1, len(subset)):
                vec2 = subset[j]
                dist = np.linalg.norm(vec2 - vec1)
                
                if dist < min:
                    min = dist
                    min_pair = (i, j)
                if dist > max:
                    max = dist
                    max_pair = (i, j)

                distances.append(dist)
                # print(f'({i}, {j}): {dist:.3f}')

        print(f'{lang}: ')
        print(f'mean = {np.mean(distances)}')
        print(f'median = {np.median(distances)}')
        print(f'min = {min} between {min_pair}')
        print(f'max = {max} between {max_pair}')

    for id in range(0, 20):
        subset = embeddings[data['sent_id'] == id]

        distances = []
        min = 1000000
        min_pair = None
        max = 0
        max_pair = None
        for i in range(0, len(subset)):
            vec1 = subset[i]
            for j in range(i+1, len(subset)):
                vec2 = subset[j]
                dist = np.linalg.norm(vec2 - vec1)
                
                if dist < min:
                    min = dist
                    min_pair = (i, j)
                if dist > max:
                    max = dist
                    max_pair = (i, j)

                distances.append(dist)
                # print(f'({i}, {j}): {dist:.3f}')

        print(f'{id}: ')
        print(f'mean = {np.mean(distances)}')
        print(f'median = {np.median(distances)}')
        print(f'min = {min} between {min_pair}')
        print(f'max = {max} between {max_pair}')
    # with open(f'{output_dir}/distances.txt', 'a') as file:
        # file.write(f'{subset_name} has average pairwise distance of ')

def plot_dbscan_clusters(embeddings, output_dir, figsize = (15, 10), dpi=300):
        
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    for epsilon in range(1, 11):
        for j in range(2, 6):
            
            e = epsilon / 10.0 + 2

            dbscan = DBSCAN(eps=(e), min_samples=j)  
            clusters = dbscan.fit_predict(embeddings)

            plt.figure(figsize=figsize)

            # Plot the clusters
            unique_clusters = np.unique(clusters)
            print(f'number of clusters: {unique_clusters} -- ({e}, {j})')

            colors = sns.color_palette('husl', n_colors=len(unique_clusters))

            for i, cluster in enumerate(unique_clusters):
                if cluster == -1:
                    # Noise points in black
                    mask = clusters == cluster
                    plt.scatter(
                        embeddings_2d[mask, 0],
                        embeddings_2d[mask, 1], 
                        color='black',
                        label='Noise',
                        s=100
                        )
                else:

                    mask = clusters == cluster
                    cluster_points = embeddings_2d[mask]
                    
                    # Plot the points
                    plt.scatter(
                        embeddings_2d[mask, 0],
                        embeddings_2d[mask, 1],
                        color=colors[i % len(colors)],
                        label=f'Cluster {cluster}',
                        alpha=0.8,
                        s=120
                        )
                    
                    # Add convex hull boundary
                    if len(cluster_points) > 2:  # Need at least 3 points for a hull
                        hull = ConvexHull(cluster_points)
                        hull_vertices = cluster_points[hull.vertices]
                        hull_vertices = np.vstack((hull_vertices, hull_vertices[0]))
                        # Plot the hull boundary
                        plt.plot(hull_vertices[:, 0], hull_vertices[:, 1], 
                                c=colors[i % len(colors)], linestyle='--', alpha=0.5)


            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.title(f'DBSCAN Clustering Results -- epsilon = 3, min_samples = {i}')
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'dbscan_e{epsilon}_ms{j}.png'),
                dpi=dpi,
                bbox_inches='tight'
            )
            plt.close()

def main(
    model,
    tokenizer,
    dir_tag,
    csv_path,
    languages,
    plot_dir='plots',
    sent_id_range: Optional[Tuple[int, int]] = None,
    sample_size: Optional[int] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    base_model = None
):

    num_sents = sent_id_range[1] - sent_id_range[0] + 1
    output_dir = f'{plot_dir}/{num_sents}_sents-{dir_tag}'

    # if os.path.exists(output_dir):
    #     print(f'\ndirectory "{output_dir}" already exists')
    #     if input('overwrite existing directory? [y/n] ') != 'y':
    #         print('exiting...')
    #         exit()

    print("Loading and filtering data...")

    data = load_and_filter_data(
        csv_path,
        languages=languages,
        sent_id_range=sent_id_range,
        sample_size=sample_size
    )
    

    print(f"Analyzing {len(data)} sentences...")
    embeddings = analyze_sentences(model, tokenizer, data, languages)
   
    reduced_embeddings = get_reduction(embeddings, perplexity=perplexity, n_iter=n_iter)

    print("Creating visualizations...")
    visualize_embeddings(
        data,
        reduced_embeddings,
        output_dir=output_dir,
    )

    lower = min(num_sents, len(languages))
    n_clusters_range = range(int(lower*0.8), max(int(lower*1.5), 8))
    best_silhouette = (-1, 0)
    for n_clusters in n_clusters_range:

        labels, cluster_scores, score = perform_hierarchical_clustering(embeddings, n_clusters=n_clusters, output_dir=output_dir, dpi=300)
        if score > best_silhouette[1]:
            plot_clustering_results(data, labels, reduced_embeddings, output_dir, tag=n_clusters, scores=cluster_scores)
            best_silhouette = (n_clusters, score)
        # print(f'n_clusters: {n_clusters}')
        # print(f'avg silhouette score: {score}\n')
    
    labels, cluster_scores, score = perform_hierarchical_clustering(embeddings, n_clusters=1000, output_dir=output_dir, dpi=300)
    print(f'average score with 1000 clusters: {score}')

    print(f'Using {best_silhouette[0]} clusters resulted in the highest silhouette score: {best_silhouette[1]:.3f}')

    # plot_dbscan_clusters(embeddings, output_dir)

    angle_data = get_cosine_similarity(embeddings, data, languages)
    plot_cosine_similarity(angle_data, output_dir)

    # put info about sentences and model in txt file 
    with open(f'{output_dir}/info.txt', 'w') as file:
        
        file.write(f'base model used: {base_model}\n\n')
        file.write('Sentences used:\n')
        for i, row in data[data['language'] == 'eng'].iterrows():
            file.write(f"{row['sent_id']}:\t{row['text']}\n")

# def find_best_n_clusters(embeddings, reduced_embeddings, num_sents):
#     for i in range(int(num_sents * 0.8), )


if __name__ == "__main__":

    base_model = "facebook/nllb-200-distilled-600M"

    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    replacement_model = CustomM2M100Model(model.model.config)
    replacement_model.load_state_dict(model.model.state_dict())
    model.model = replacement_model
    
    LR_italic = ['fur_Latn', 'lij_Latn', 'lmo_Latn', 'scn_Latn', 'srd_Latn', 'vec_Latn']
    LR_indic = ['bho_Deva', 'hne_Deva', 'kas_Deva', 'mag_Deva']

    HR_indic = ['hin_Deva', 'ben_Beng'] # apparently we don't have these in seed
    HR_italic = ['cat_Latn', 'ita_Latn', 'spa_Latn'] # or these

    seed_indic = ['bho_Deva', 'hne_Deva', 'kas_Deva', 'mag_Deva', 'eng_Latn']
    seed_italic = ['fur_Latn', 'lij_Latn', 'lmo_Latn', 'scn_Latn', 'srd_Latn', 'vec_Latn', 'eng_Latn']

    main(
        model,
        tokenizer,
        dir_tag='italic',
        csv_path=NLLB_SEED_CSV,
        languages=seed_italic,   
        sent_id_range=(0, 6000),  # Optional: range of sentence IDs
        perplexity=30.0,
        n_iter=1000,
        base_model=base_model
    )


# for 1000-italic: average score with 1000 clusters: 0.1637 - way better