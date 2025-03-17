from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import sys
import json
import shutil
import evaluate                                                     # type: ignore
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from finetuning.multilingualdata import MultilingualCorpus          # type: ignore

def translate(
    text, tokenizer, model, 
    src_lang, tgt_lang, 
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):
    model.eval() # turn off training mode
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
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
    print('result:', result)
    return tokenizer.batch_decode(result, skip_special_tokens=True)

def tokenize(sents, lang, tokenizer, max_length, alt_pad_token=None):
    tokenizer.src_lang = lang
    tokens = tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    if alt_pad_token is not None:
        tokens.input_ids[tokens.input_ids == tokenizer.pad_token_id] = alt_pad_token  # e.g., -100 is a magic value ignored 
                                                                                      # in the loss function because we don't want the model to learn to predict padding ids
    return tokens

def see_tokens(sents, lang):
    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokens = tokenize(sents, lang, tokenizer, 128)
    print(tokens.input_ids)

    # token strings
    for sentence in sents:
        print(f"Sentence: {sentence}")
        print("Tokens:", [tokenizer.decode(token_id) for token_id in tokens.input_ids[sents.index(sentence)]])

    # print(tokens.attention_mask)

    

if __name__ == '__main__':
    base_model = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, clean_up_tokenization_spaces=False)

    # print(model)

    en_sents = ['Happy birthday!']
    sp_sents = ['¡Feliz cumpleaños!']

    translate(en_sents, tokenizer, model, 'eng_Latn', 'spa_Latn')

    # see_tokens(en_sents, 'eng_Latn')
    # see_tokens(sp_sents, 'spa_Latn')
    print()