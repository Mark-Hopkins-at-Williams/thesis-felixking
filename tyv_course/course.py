import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
from tqdm.auto import tqdm, trange
from remove_unk import preproc

trans_df = pd.read_csv('./rus_tyv_parallel_50k.tsv', sep="\t")

df_train = trans_df[trans_df.split=='train'].copy() # 49000 items
df_dev = trans_df[trans_df.split=='dev'].copy()     # 500 items
df_test = trans_df[trans_df.split=='test'].copy()   # 500 items

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.src_lang = "rus_Cyrl"
inputs = tokenizer(text="поля озарились утренним солнцем", return_tensors="pt")
translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn")
)
print(tokenizer.decode(translated_tokens[0], skip_special_tokens=True))
# The fields were lit by the morning sun

def word_tokenize(text):
    """
    Split a text into words, numbers, and punctuation marks
    (for languages where words are separated by spaces)
    """
    return re.findall('(\w+|[^\w\s])', text)

smpl = df_train.sample(10000, random_state=1)
smpl['rus_toks'] = smpl.ru.apply(tokenizer.tokenize)
smpl['tyv_toks'] = smpl.tyv.apply(tokenizer.tokenize)
smpl['rus_words'] = smpl.ru.apply(word_tokenize)
smpl['tyv_words'] = smpl.tyv.apply(word_tokenize)


texts_with_unk = [
    text for text in tqdm(trans_df.tyv) 
    if tokenizer.unk_token_id in tokenizer(text).input_ids
]
print(len(texts_with_unk))
# 163

texts_with_unk_normed = [
    text for text in tqdm(texts_with_unk) 
    if tokenizer.unk_token_id in tokenizer(preproc(text)).input_ids
]
print(len(texts_with_unk_normed))  # 0
