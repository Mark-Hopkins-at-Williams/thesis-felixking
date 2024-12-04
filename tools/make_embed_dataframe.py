import pandas as pd
from bottle import CustomM2M100Model
from plot_encoder_states import analyze_sentences
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from configure import NLLB_SEED_CSV, NLLB_SEED_LANGS, SEED_EMBED_PARQUET

if __name__ == '__main__':

    base_model = "facebook/nllb-200-distilled-600M"

    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    replacement_model = CustomM2M100Model(model.model.config)
    replacement_model.load_state_dict(model.model.state_dict())
    model.model = replacement_model

    df = pd.read_csv(NLLB_SEED_CSV)

    # only use the train rows, ie from seed
    df = df[df['split'] == 'train']

    # get embeddings        
    embeddings = analyze_sentences(model, tokenizer, df, NLLB_SEED_LANGS)
    df['embedding'] = list(embeddings)

    # don't need split
    df = df.drop('split', axis=1)

    # write to parquet file to preserve embeddings
    df.to_parquet(SEED_EMBED_PARQUET, index=False)


