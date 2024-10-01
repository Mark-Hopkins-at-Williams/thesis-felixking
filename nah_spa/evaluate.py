# BLEU = 6.50 28.8/9.0/3.7/1.9 (BP = 1.000 ratio = 1.038 hyp_len = 7524 ref_len = 7247)
# chrF2++ = 23.78

"""
NLLB-1.3B (single best)         30.1
1.3B random initialisation      24.3
NLLB-1.3B + bibles              28.0
"""


from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utility import get_def_train, get_def_dev
from transformers import NllbTokenizer


model_load_name = '/mnt/storage/fking/models/nllb-nah-spa-v1'
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
model_tok_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_tok_name)


def translate(
    text, src_lang='gug_Latn', tgt_lang='spa_Latn', 
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

import sacrebleu
import pandas as pd
bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

df_train = get_def_train()
df_dev = get_def_dev()

spa_translations = batched_translate(df_dev['nah'].tolist(), src_lang='gug_Latn', tgt_lang='spa_Latn')

print(bleu_calc.corpus_score(spa_translations, [df_dev['spa'].tolist()]))
print(chrf_calc.corpus_score(spa_translations, [df_dev['spa'].tolist()]))