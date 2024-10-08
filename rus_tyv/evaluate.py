import os
import sys
import sacrebleu
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

results_path = "/mnt/storage/fking/thesis-felixking/results/rus-tyv/results.txt"
model_load_name = '/mnt/storage/fking/models/'
model_name = ""

# Parse input and make sure it makes sense
if len(sys.argv) < 2:
    print("usage: python3 evaluate.py <model dir>")
    exit()
else:
    model_name = sys.argv[1]
    model_load_name += model_name

    if not os.path.exists(model_load_name):
        print("model path does not exist")
        exit()



model_tok_name = "facebook/nllb-200-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_tok_name)


def translate(
    text, src_lang='kir_Cyrl', tgt_lang='rus_Cyrl', 
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):
    """Translates a string or a list of strings."""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    model.eval() # turn off training mode
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True, 
        max_length=max_input_length
    )
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)
    

def batched_translate(texts, batch_size=16, **kwargs):
    """Translate texts in batches of similar length"""
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in tqdm(range(0, len(texts2), batch_size)):
        results.extend(translate(texts2[i: i+batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]

bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++


larger_file = '/mnt/storage/fking/data/rus_tyv_123k.csv'
df = pd.read_csv(larger_file, sep=";")
df.dropna(subset=['ru', 'tyv'], inplace=True)

df_train, df_devtest = train_test_split(df, test_size=1000, random_state=1)
df_dev_123k, df_test = train_test_split(df_devtest, test_size=0.5, random_state=1)

rus_translations_123k = batched_translate(df_dev_123k['tyv'].tolist(), src_lang='kir_Cyrl', tgt_lang='rus_Cyrl')
bleu_result_123k = bleu_calc.corpus_score(rus_translations_123k, [df_dev_123k['ru'].tolist()])
chrf_result_123k = chrf_calc.corpus_score(rus_translations_123k, [df_dev_123k['ru'].tolist()])


# trans_df = pd.read_csv('/mnt/storage/hopkins/thesis/data/rus_tyv_parallel_50k.tsv', sep="\t")
# df_dev_50k = trans_df[trans_df.split=='dev'].copy()     # 500 items

# rus_translations_50k = batched_translate(df_dev_50k['tyv'].tolist(), src_lang='kir_Cyrl', tgt_lang='rus_Cyrl')
# bleu_result_50k = bleu_calc.corpus_score(rus_translations_50k, [df_dev_50k['ru'].tolist()])
# chrf_result_50k = chrf_calc.corpus_score(rus_translations_50k, [df_dev_50k['ru'].tolist()])

now = datetime.now()
# mm/dd/YY H:M:S
date_time = now.strftime("%m/%d/%Y %H:%M:%S")

sep = " --- "

# output = f"{model_name}{sep}{date_time}\n123k:\n{bleu_result_123k}\n{chrf_result_123k}\n\n50k:\n{bleu_result_50k}\n{chrf_result_50k}\n\n"
output = f"{model_name}{sep}123k{sep}{date_time}\n{bleu_result_123k}\n{chrf_result_123k}\n\n"

if os.path.exists(results_path):
    with open(results_path, 'a') as file:

        file.write(output)
else:
    # File does not exist, create it
    with open(results_path, 'w') as file:

        file.write(output)

