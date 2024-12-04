import pandas as pd
trans_df = pd.read_csv('rus_tyv_parallel_50k.tsv', sep="\t")


df_train = trans_df[trans_df.split=='train'].copy() # 49000 items
df_dev = trans_df[trans_df.split=='dev'].copy()     # 500 items
df_test = trans_df[trans_df.split=='test'].copy()   # 500 items

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.src_lang = "rus_Cyrl"
inputs = tokenizer(text="поля озарились утренним солнцем", return_tensors="pt")
print(inputs)
#translated_tokens = model.generate(
#    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
#)
#print(tokenizer.decode(translated_tokens[0], skip_special_tokens=True))
# The fields were lit by the morning sun