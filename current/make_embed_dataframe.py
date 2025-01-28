import pandas as pd
from bottle import CustomM2M100Model
from plot_encoder_states import analyze_sentences
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main(csv_path, pkl_path, langs, split=False, NLLB_model_size='600M'):
    base_model = f"facebook/nllb-200-distilled-{NLLB_model_size}"

    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    replacement_model = CustomM2M100Model(model.model.config)
    replacement_model.load_state_dict(model.model.state_dict())
    model.model = replacement_model

    print('reading csv...')
    df = pd.read_csv(csv_path)

    if split:
        # only use the train rows, ie from seed
        df = df[df['split'] == 'train']

    print('getting embeddings...')
    embeddings = analyze_sentences(model, tokenizer, df, langs, batch_size=16)

    df['embedding'] = list(embeddings)

    if split:
        # don't need split
        df = df.drop('split', axis=1)

    print('writing df to file')
    # write to parquet file to preserve embeddings
    df.to_pickle(pkl_path)


ep_langs = [
    'bul_Cyrl', 'ces_Latn', 'dan_Latn', 'deu_Latn', 'ell_Grek', 'eng_Latn',
    'spa_Latn', 'est_Latn', 'fin_Latn', 'fra_Latn', 'hun_Latn', 'ita_Latn',
    'lit_Latn', 'lvs_Latn', 'nld_Latn', 'pol_Latn', 'por_Latn', 'ron_Latn',
    'slk_Latn', 'slv_Latn', 'swe_Latn'
]
seed_langs = [
    'pbt_Arab', 'bho_Deva', 'nus_Latn', 'ban_Latn', 'dzo_Tibt', 'mni_Beng', 'lim_Latn', 
    'ltg_Latn', 'ace_Latn', 'crh_Latn', 'srd_Latn', 'taq_Latn', 'mri_Latn', 'ary_Arab', 
    'bam_Latn', 'knc_Arab', 'eng_Latn', 'knc_Latn', 'dik_Latn', 'prs_Arab', 'bjn_Arab', 
    'vec_Latn', 'fur_Latn', 'kas_Deva', 'kas_Arab', 'arz_Arab', 'lij_Latn', 'ace_Arab', 
    'bjn_Latn', 'scn_Latn', 'bug_Latn', 'lmo_Latn', 'szl_Latn', 'hne_Deva', 'fuv_Latn', 
    'taq_Tfng', 'shn_Mymr', 'mag_Deva']

if __name__ == '__main__':
    main(
    '/mnt/storage/hopkins/data/nllb/seed/nllb_seed.csv', 
    '/mnt/storage/fking/data/seed/seed_1.3B.pkl', 
    seed_langs, 
    split=True,
    NLLB_model_size='1.3B'
    )

    # main(
    # '/mnt/storage/fking/data/europarl/europarl_10k.csv', 
    # '/mnt/storage/fking/data/europarl/europarl_10k_1.3B.pkl', 
    # ep_langs, 
    # split=True,
    # NLLB_model_size='1.3B'
    # )
        
