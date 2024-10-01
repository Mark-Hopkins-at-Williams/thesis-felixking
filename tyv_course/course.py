import re
import gc
import os
import json
import torch
import shutil
import random
import numpy as np
import pandas as pd
import sentencepiece as spm
from typing import List, Tuple
from remove_unk import preproc
from collections import Counter
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers.optimization import Adafactor
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer, get_constant_schedule_with_warmup


# constants
LANGS = [('ru', 'rus_Cyrl'), ('tyv', 'kir_Cyrl')]
model_name = 'facebook/nllb-200-distilled-600M'
tsv_file = './rus_tyv_parallel_50k.tsv'
batch_size = 16  # 32 already doesn't fit well to 15GB of GPU memory
max_length = 128
warmup_steps = 1_000
training_steps = 57000
MODEL_SAVE_PATH = '../../models/nllb-rus-tyv-v1'



trans_df = pd.read_csv(tsv_file, sep="\t")
df_train = trans_df[trans_df.split=='train'].copy() # 49000 items
df_dev = trans_df[trans_df.split=='dev'].copy()     # 500 items
df_test = trans_df[trans_df.split=='test'].copy()   # 500 items
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3,
)

losses = []
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

def get_batch_pairs(batch_size, data=df_train):
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    return xx, yy, long1, long2

model.train()
x, y, loss = None, None, None
cleanup()

for i in tqdm(range(len(losses), training_steps)):
    xx, yy, lang1, lang2 = get_batch_pairs(batch_size)
    try:
        tokenizer.src_lang = lang1
        x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
        tokenizer.src_lang = lang2
        y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
        y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

        loss = model(**x, labels=y.input_ids).loss
        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    except RuntimeError as e:
        optimizer.zero_grad(set_to_none=True)
        x, y, loss = None, None, None
        cleanup()
        print('error', max(len(s) for s in xx + yy), e)
        continue

    if i % 1000 == 0:
        print(i, np.mean(losses[-1000:]))

    if i % 1000 == 0 and i > 0:
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)

pd.Series(losses).ewm(100).mean().plot()