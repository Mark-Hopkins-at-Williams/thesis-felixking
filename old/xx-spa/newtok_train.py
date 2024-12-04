import gc
import os
import sys
import torch
import random
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from preprocess import preproc
from transformers import NllbTokenizer
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# TODO
# add -e command line option to evaluate at the end

codes = { # these are just the codes that americasnlp uses for their files
    "ashaninka": "cni",
    "bribri": "bzd", 
    "guarani": "gn",   
    "quechua": "quy",  
    "aymara": "aym",   
    "shipibo_konibo": "shp",
    "chatino": "ctp",
    "hñähñu": "oto",
    "nahuatl": "nah",
    "raramuri": "tar",
    "wixarika": "hch"  
    }

size = "1.3B"       #default size
batch_size = 16     # 32 already doesn't fit well to 15GB of GPU memory
max_length = 128    # token sequences will be truncated
training_steps = 60000  
MODEL_SAVE_PATH = '/mnt/storage/fking/models/' 
NEW_SPM_PATH = "/mnt/storage/fking/models/toks/spm_nllb_quechua_tok.model"
csv_file = '/mnt/storage/fking/americasnlp2024/ST1_MachineTranslation/data/'
dev_losses = []     # with this list, I do very simple tracking of average loss
train_losses = []   # with this list, I do very simple tracking of average loss
src_lang = ""
evaluate = False    # do not run evaluate after finishing training

def get_batch_pairs(batch_size, data):
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    return xx, yy, long1, long2

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("usage: python3 train.py <src-tgt> <model name> <¿size?>")
        exit()
    else:
        csv_file += sys.argv[1]
        csv_file += "/all.csv"

        src_lang = codes[sys.argv[1].split("-")[0]]

        MODEL_SAVE_PATH += sys.argv[2]

        if not os.path.exists(csv_file):
            print("src-tgt directory does not exist")
            exit()
        if os.path.exists(MODEL_SAVE_PATH):
            print("model directory already exists")
            exit()

        for i in range(3, len(sys.argv)):
            match sys.argv[i]:
                case x if x in ["s", 'small', '600M', '600m']:
                    size = "600M"
                case x if x in ['m', 'medium', '1.3B', '1.3b']:
                    size = "1.3B"
                case "-e":
                    evaluate = True
                case _:
                    print("optional arguments not recognized")
                    # print("accepted sizes are \t[s, small,  600M, 600M] for nllb-600M or \n\t\t\t[m, medium, 1.3B, 1.3b] for nllb-1.3B")
                
    model_name = "facebook/nllb-200-distilled-" + size
    src_lang_code = codes[sys.argv[1].split("-")[0]] # e.g. "nah" or "quy"

    now = datetime.now()
    # mm/dd/YY H:M:S
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")

    proxy_code = 'ayr_Latn'

    # basic info shows up at the top of log_<pid>.out
    print(f"\n{model_name}\n{sys.argv[1]}\n{MODEL_SAVE_PATH}\n{date_time}\nusing {proxy_code} language code\n")


    trans_df = pd.read_csv(csv_file, sep=",")

    df_train = trans_df[trans_df.split=='train'].copy() 
    df_dev = trans_df[trans_df.split=='dev'].copy()     
    df_test = trans_df[trans_df.split=='test'].copy()   


    # loading the tokenizers
    tokenizer_old = NllbTokenizer.from_pretrained(model_name)
    tokenizer = NllbTokenizer.from_pretrained(model_name, vocab_file=NEW_SPM_PATH)
    print(len(tokenizer_old), len(tokenizer)) # 256204, 268559
    added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
    print(len(added_vocab))  # 12355


    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # re-initializing the new embeddings
    for t in tqdm(added_vocab):
        tt = tokenizer_old(t, add_special_tokens=False).input_ids
        if len(tt) == 0:
            tt = [tokenizer_old.unk_token_id]
        idx = tokenizer.convert_tokens_to_ids(t)
        model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)



    model.cuda()
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    LANGS = [('spa', 'spa_Latn'), (src_lang_code, proxy_code)]
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

    x, y, train_loss = None, None, None
    x_dev, y_dev, dev_loss = None, None, None
    best_dev_loss = None
    last_best = 0
    patience = 30000
    cleanup()

    for i in tqdm(range(len(train_losses), training_steps)):
        xx, yy, lang1, lang2 = get_batch_pairs(batch_size, df_train)
        xx_dev, yy_dev, lang1_dev, lang2_dev = get_batch_pairs(batch_size, data=df_dev)

        try:
            model.train()
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            x_dev = tokenizer(xx_dev, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            tokenizer.src_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            y_dev = tokenizer(yy_dev, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            # -100 is a magic value ignored in the loss function
            # because we don't want the model to learn to predict padding ids
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
            y_dev.input_ids[y_dev.input_ids == tokenizer.pad_token_id] = -100

            train_loss = model(**x, labels=y.input_ids).loss
            train_loss.backward()
            train_losses.append(train_loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            with torch.no_grad():
                model.eval()
                dev_loss = model(**x_dev, labels=y_dev.input_ids).loss
                dev_losses.append(dev_loss.item())

        except RuntimeError as e:  # usually, it is out-of-memory
            optimizer.zero_grad(set_to_none=True)
            x, y, train_loss = None, None, None
            x_dev, y_dev, dev_loss = None, None, None
            cleanup()
            print('error', max(len(s) for s in xx + yy), e)
            continue

        if i % 1000 == 0:
            # each 1000 steps, I report average loss at these steps
            print(f'step {i} (train): {np.mean(train_losses[-1000:])}')
            print(f'step {i} (dev):   {np.mean(dev_losses[-1000:])}')
            sys.stdout.flush()

        if i % 1000 == 0 and i > 0 and (best_dev_loss is None or dev_loss < best_dev_loss):
            print("Saving new best model!")
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            best_dev_loss = dev_loss
            last_best = i
        
        if i - last_best >= patience:
            break
        
    if evaluate:
        evaluate.main(sys.argv[:-1])