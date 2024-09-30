import pandas as pd
import sys
from transformers.optimization import Adafactor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_constant_schedule_with_warmup
from preprocess import preproc
import random
import gc
import torch
from tqdm import tqdm
import numpy as np

model_name = "facebook/nllb-200-distilled-600M"
tsv_file = '/mnt/storage/hopkins/thesis/data/rus_tyv_parallel_50k.tsv'
batch_size = 16  # 32 already doesn't fit well to 15GB of GPU memory
max_length = 128  # token sequences will be truncated
training_steps = 60000  



trans_df = pd.read_csv(tsv_file, sep="\t")
df_train = trans_df[trans_df.split=='train'].copy() # 49000 items
df_dev = trans_df[trans_df.split=='dev'].copy()     # 500 items
df_test = trans_df[trans_df.split=='test'].copy()   # 500 items

from datasets import load_dataset
from collections import Counter

tyv_wiki = load_dataset("graelo/wikipedia", "20230601.tyv")
tyv_wiki
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'url', 'title', 'text'],
#         num_rows: 3459
#     })
# })
print(sum(len(t) for t in tyv_wiki['train']['text']))  # 7568832
print(sum(len(t) for t in trans_df.tyv.dropna()))      # 3573803

all_texts = tyv_wiki['train']['text'] + df_train.tyv.dropna().tolist()
all_text_normalized = [preproc(t) for t in tqdm(all_texts)]
chars_cnt = Counter(c for t in all_text_normalized for c in t)
required_chars = ''.join([
    k for k, v in chars_cnt.most_common() 
    if v >= 3 and k not in ' '
])

import sentencepiece as spm
all_texts_file = 'myv_texts_plain.txt'
SPM_PREFIX = 'spm_tyvan_16k'
with open(all_texts_file, 'w') as f:
    for i, text in enumerate(all_texts):
        print(text, file=f)

spm.SentencePieceTrainer.train(
    input=all_texts_file,
    model_prefix=SPM_PREFIX,
    vocab_size=2**14,  # 16K
    character_coverage = 1,
    num_threads=16,
    train_extremely_large_corpus=False,
    add_dummy_prefix=False,
    max_sentencepiece_length=128,
    max_sentence_length=4192*4,
    pad_id=0,
    eos_id=1,
    unk_id=2,
    bos_id=-1,
    required_chars=required_chars,
)

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
# At this step, the code may throw an error about protobuf. Do as it tells.
from transformers import NllbTokenizer

# reading the NLLB and the Tyvan sentencepiece models into a native format
tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
sp_trained = spm.SentencePieceProcessor(model_file=f'{SPM_PREFIX}.model')
added_spm = sp_pb2_model.ModelProto()
added_spm.ParseFromString(sp_trained.serialized_model_proto())
old_spm = sp_pb2_model.ModelProto()
old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

# adding the missing tokens to the NLLB sentencepiece model
nllb_tokens_set = {p.piece for p in old_spm.pieces}
prev_min_score = old_spm.pieces[-1].score
for p in added_spm.pieces:
    piece = p.piece
    # !!! THIS FIX WAS ADDED LATER; it is required for CT2 compatibility !!!
    # 1 is ordinary token, non-1 is special token; we don't want to copy the special tokens
    if p.type != 1:
        continue
    if piece not in nllb_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        # for all new tokens, I'll set a lower score (priority)
        new_p.score = p.score + prev_min_score
        old_spm.pieces.append(new_p)

# saving the result to disk
NEW_SPM_NAME = 'spm_nllb_tyvan_268k.model'
with open(NEW_SPM_NAME, 'wb') as f:
    f.write(old_spm.SerializeToString())
