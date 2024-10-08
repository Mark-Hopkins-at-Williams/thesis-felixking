import os
import sys
import sacrebleu
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import NllbTokenizer
from utility import get_def_train, get_def_dev
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# (ex) python3 evaluate.py nahuatl-spanish nllb-nah-spa-v1

dir_name        = "/mnt/storage/fking/americasnlp2024/ST1_MachineTranslation/data/"
model_load_name = '/mnt/storage/fking/models/'
model_name = ""
model_size = ""

# Parse input and make sure it makes sense
if len(sys.argv) < 4:
    print("usage: python3 evaluate.py <src-tgt> <model dir> <model size>")
    exit()
else:
    model_name = sys.argv[2]
    dir_name        += sys.argv[1]
    model_load_name += model_name

    if not os.path.exists(dir_name):
        print("src-tgt path does not exist")
        exit()
    if not os.path.exists(model_load_name):
        print("model path does not exist")
        exit()

    match sys.argv[3]:
        case x if x in ["s", 'small', '600M', '600m']:
            size = "600M"
        case x if x in ['m', 'medium', '1.3B', '1.3b']:
            size = "1.3B"
        case _:
            print("accepted sizes are \t[s, small,  600M, 600M] for nllb-600M or \n\t\t\t[m, medium, 1.3B, 1.3b] for nllb-1.3B")

# model size doesn't affect tokenizer so 600M is fine
model_tok_name = "facebook/nllb-200-distilled-600M"

results_path = "/mnt/storage/fking/thesis-felixking/results/americas/results.txt"
lang_code = codes[sys.argv[1].split("-")[0]] # e.g. "nah" or "quy"

model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_tok_name)


def translate(
    text, src_lang='gug_Latn', tgt_lang='spa_Latn', 
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):

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

    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in tqdm(range(0, len(texts2), batch_size)):
        results.extend(translate(texts2[i: i+batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]


bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

#dev set
df_dev = get_def_dev(dir_name, lang_code)


# ugly but must be done- americas nlp codes do not quite agree with nllb ones
src_lang = "gug_Latn"
if(lang_code == "aym"):
    src_lang = "ayr_Latn"
if(lang_code == "quy"):
    src_lang = "quy_Latn"


spa_translations = batched_translate(df_dev[lang_code].tolist(), src_lang=src_lang, tgt_lang='spa_Latn')


bleu_result  = str(bleu_calc.corpus_score(spa_translations, [df_dev['spa'].tolist()]))
chrf_result = str(chrf_calc.corpus_score(spa_translations, [df_dev['spa'].tolist()]))

 
now = datetime.now()
# mm/dd/YY H:M:S
date_time = now.strftime("%m/%d/%Y %H:%M:%S")

sep = " --- "

output = f"{model_name}{sep}{size}{sep}{date_time}\n{bleu_result}\n{chrf_result}\n\n"

if os.path.exists(results_path):
    with open(results_path, 'a') as file:

        file.write(output)
else:
    # File does not exist, create it
    with open(results_path, 'w') as file:

        file.write(output)




