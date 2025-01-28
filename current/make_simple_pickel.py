import pandas as pd
from configure import NLLB_SEED_LANGS, SEED_EMBED_PICKLE, TEN_SEED_LANGS

if __name__ == '__main__':
    
    langs = NLLB_SEED_LANGS
    df = pd.read_pickle(SEED_EMBED_PICKLE)
    df = df[(df.apply(lambda row: f"{row['language']}_{row['script']}" in langs, axis=1)) & (df['sent_id'] < 50)]

    print(len(df))
    df.to_pickle('simple.pkl')