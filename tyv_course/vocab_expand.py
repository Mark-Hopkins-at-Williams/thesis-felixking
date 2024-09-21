import re
import pandas as pd
import sentencepiece as spm
from remove_unk import preproc
from collections import Counter
from datasets import load_dataset
from tqdm.auto import tqdm, trange
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

trans_df = pd.read_csv('./rus_tyv_parallel_50k.tsv', sep="\t")

df_train = trans_df[trans_df.split=='train'].copy() # 49000 items
df_dev = trans_df[trans_df.split=='dev'].copy()     # 500 items
df_test = trans_df[trans_df.split=='test'].copy()   # 500 items


tyv_wiki = load_dataset("graelo/wikipedia", "20230601.tyv", trust_remote_code=True)
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


all_texts_file = 'myv_texts_plain.txt'
SPM_PREFIX = 'spm_tyvan_16k'
with open(all_texts_file, 'w') as f:
    for i, text in enumerate(all_texts):
        print(text, file=f)


##############################################################################
###### Training the sentencepiece tokenizer the Tuvan wiki ###################
###### Database that we processed aboved #####################################
##############################################################################

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



##############################################################################
###### Performing "surgery" on the pretrained model ##########################
###### extracting the sentencepiece model from the standard NLLB #############
###### tokenizer and enriching it from all tokens from the Tyvan #############
###### tokenizer that have been missing from the NLLB tokenizer ##############
##############################################################################


from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
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

from transformers import AutoModelForSeq2SeqLM
model_name = 'facebook/nllb-200-distilled-600M'

# loading the tokenizers
tokenizer_old = NllbTokenizer.from_pretrained(model_name)
tokenizer = NllbTokenizer.from_pretrained(model_name, vocab_file=NEW_SPM_NAME)
print(len(tokenizer_old), len(tokenizer)) # 256204, 268559
added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
print(len(added_vocab))  # 12355

# loading and resizing the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# re-initializing the new embeddings
for t in tqdm(added_vocab):
    tt = tokenizer_old(t, add_special_tokens=False).input_ids
    if len(tt) == 0:
        tt = [tokenizer_old.unk_token_id]
    idx = tokenizer.convert_tokens_to_ids(t)
    model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)