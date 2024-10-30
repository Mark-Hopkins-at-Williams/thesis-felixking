import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from finetune import tokenize
from bottle import CustomM2M100Model
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
from scipy.spatial.distance import cosine
from configure import NLLB_SEED_CSV, NLLB_SEED_LANGS, TEN_SEED_LANGS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_and_filter_data(
    csv_path,
    languages = None,
    sent_id_range = None,
    sample_size = None
):

    # read
    df = pd.read_csv(csv_path)

    # filter by lang
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

def get_sentence_embedding(
    model,
    tokenizer,
    sentence,
    language,
    max_length = 128
):

    # tokenize the input
    inputs = tokenize([sentence], language, tokenizer, max_length=max_length).to(model.device)
    
    # Get encoder states
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        encoder_states = model.model.custom_module.snapshot
        
        # Get attention mask to properly average token embeddings
        attention_mask = inputs.attention_mask
        
        # Average the token embeddings, excluding padding tokens
        masked_states = encoder_states * attention_mask.unsqueeze(-1)
        sum_states = masked_states.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True)
        sentence_embedding = (sum_states / sum_mask).cpu().numpy()[0]
        
    return sentence_embedding

def analyze_sentences(
    model,
    tokenizer,
    sentence_data,
    batch_size = 32
):
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    embeddings = []
    metadata = []   # unclear if this is at all necessary - I guess it keeps the 
                    # order of the data and the embeddings the same? but I think that already happens
                    # also do batches need to be in the same language?
    
    # Process sentences in batches
    for i in tqdm(range(0, len(sentence_data), batch_size)):

        # this scheme is also kind of dumb - not really taking advantage of batch translation
        batch = sentence_data.iloc[i:i+batch_size]
        # print(f"Processing batch {i//batch_size + 1}/{len(sentence_data)//batch_size + 1}")
        
        for _, row in batch.iterrows():
            embedding = get_sentence_embedding(
                model,
                tokenizer,
                row['text'],
                row['language']
            )
            
            embeddings.append(embedding)
            metadata.append({
                'language': row['language'],
                'script': row['script'],
                'sent_id': row['sent_id'],
                'text': row['text'],
                'split': row['split']
            })
    
    return np.array(embeddings), pd.DataFrame(metadata)

def visualize_embeddings(
    embeddings,
    metadata,
    output_dir = 'plots',
    filename_prefix = 'sentence_embeddings',
    method = 'tsne',
    perplexity = 30.0,
    n_iter = 1000,
    random_state = 42, #for reproducibility, maybe don't need this
    dpi = 300,
    figsize = (15, 10)
):

    os.makedirs(output_dir, exist_ok=True)

    n_samples = len(embeddings)
    
    # Automatically adjust perplexity based on dataset size
    if method == 'tsne':
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
        method_name = f'TSNE (perplexity={perplexity})'
    else:
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
        method_name = f'PCA (variance={reducer.explained_variance_ratio_})'
    

    # Create color palette for languages
    unique_languages = metadata['language'] + '_' + metadata['script']
    unique_languages = sorted(unique_languages.unique())

    color_palette = sns.color_palette('husl', n_colors=len(unique_languages))
    color_dict = dict(zip(unique_languages, color_palette))

    # Cycle through markers for languages
    # markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '8']  # Different marker shapes
    markers = ['o', 's', '^', 'x']  # Different marker shapes
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
    
    plt.title(f'Sentence Embeddings by Language ({method_name})')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(output_dir, f'{filename_prefix}_{method}_by_lang.png'),
        dpi=dpi,
        bbox_inches='tight'
    )
    plt.close()
    
    sent_ids = metadata['sent_id'].unique()

    if len(sent_ids <= 20):
    
        color_palette = sns.color_palette('husl', n_colors=len(sent_ids))
        color_dict = dict(zip(sent_ids, color_palette))
        # Plot by sentence ID (to see parallel sentences)
        
        plt.figure(figsize=figsize)

        for id in sent_ids:
            mask = metadata['sent_id'] == id
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                label=id,
                color=color_dict[id],
                marker=markers[int(id) % len(markers)],
                alpha=0.7,
                s=100
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=metadata['sent_id'],
            cmap='viridis',
            alpha=0.7,
            s=100
        )
    
        plt.colorbar(scatter, label='Sentence ID')

    plt.title(f'Sentence Embeddings by Sentence ID ({method_name})')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(output_dir, f'{filename_prefix}_{method}_by_id.png'),
        dpi=dpi,
        bbox_inches='tight'
    )
    plt.close()

def get_subset_angle_mean(subset):
    length = subset.shape[0]
    angles = []
    for i in range(0, length):
        embedding1 = subset[i]
        for j in range(i + 1, length):
            embedding2 = subset[j]
            angle = 1 - cosine(embedding1, embedding2)
            angles.append(angle)
        
    return np.mean(angles)

def get_disjoint_set_angle_mean(set1, set2):
    len1 = set1.shape[0]
    len2 = set2.shape[0]
    angles = []
    for i in range(0, len1):
        embedding1 = set1[i]
        for j in range(0, len2):
            embedding2 = set2[j]
            angle = 1 - cosine(embedding1, embedding2)
            angles.append(angle)
        
    return np.mean(angles)

def get_cosine_similarity(embeddings, data, langs):

    # calculate average angle between embeddings by language
    lang_means = []
    print("Calculating cosine similarities between sentences in the same language...")
    for lang in langs:
        mask = (data['language'] == lang.split("_")[0]) & (data['script'] == lang.split("_")[1])
        subset = embeddings[mask]
        subset_complement = embeddings[~mask]

        subset_mean = get_subset_angle_mean(subset)
        complement_mean = get_disjoint_set_angle_mean(subset, subset_complement)
        lang_means.append(subset_mean)
        print(f"\t- mean angle between embeddings in {lang}   is {subset_mean:.3f}")
        print(f"\t- mean angle between embeddings in complement is {subset_mean:.3f}")
    print(f'mean of means across all represented languages is {np.mean(lang_means)}\n')

    # calculate average angle between embeddings by sentence id
    sent_ids = data['sent_id'].unique()
    id_means = []
    print("Calculating cosine similarities between sentences with the same id...")
    for id in sent_ids:
        mask = data['sent_id'] == id
        subset = embeddings[mask]

        mean = get_subset_angle_mean(subset)
        id_means.append(mean)
        print(f"\t- mean angle between embeddings of sentence {id} is {mean:.3f}")
    print(f'mean of means across all sentence ids is {np.mean(lang_means):.3f}\n') # not sure this means much...


def main(
    model,
    tokenizer,
    csv_path,
    output_dir,
    languages,
    sent_id_range: Optional[Tuple[int, int]] = None,
    sample_size: Optional[int] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000
):

    # Load and filter data
    print("Loading and filtering data...")

    data = load_and_filter_data(
        csv_path,
        languages=languages,
        sent_id_range=sent_id_range,
        sample_size=sample_size
    )
    
    print(f"Analyzing {len(data)} sentences...")
    embeddings, data = analyze_sentences(model, tokenizer, data)
    
    get_cosine_similarity(embeddings, data, languages)

    # print("Creating visualizations...")
    # visualize_embeddings(
    #     embeddings,
    #     data,
    #     output_dir=output_dir,
    #     method='tsne',
    #     perplexity=perplexity,
    #     n_iter=n_iter
    # )
    
    print("Analysis complete!")


if __name__ == "__main__":

    base_model = "facebook/nllb-200-distilled-600M"

    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    replacement_model = CustomM2M100Model(model.model.config)
    replacement_model.load_state_dict(model.model.state_dict())
    model.model = replacement_model 

    main(
        model,
        tokenizer,
        csv_path=NLLB_SEED_CSV,
        languages=TEN_SEED_LANGS,   
        output_dir='plots',
        sent_id_range=(0, 9),  # Optional: range of sentence IDs
        perplexity=30.0,
        n_iter=1000
    )


