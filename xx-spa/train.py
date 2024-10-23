import gc
import os
import sys
import torch
import random
import evaluate
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from preprocess import preproc
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from americasnlp import AMERICASNLP_CODES, SEED_CODES


csv_file = '/mnt/storage/fking/americasnlp2024/ST1_MachineTranslation/data/'
model_save_path = '/mnt/storage/fking/models/' 
size = "1.3B"       # default size
batch_size = 16     # 32 already doesn't fit well to 15GB of GPU memory
max_length = 128    # token sequences will be truncated
training_steps = 60000  
dev_losses = []     
train_losses = []   # these lists track of average loss
evaluate = False    # do not run evaluate after finishing training

def sample_langs(source, target, lang_dict):

    pairs = list(lang_dict.items())
    opts = [("", ""), ("", "")]

    if source == "xx" or source == "yy":
        opt1 = (target, lang_dict[target]) if target in lang_dict else ("", "")
        opts[0] = random.choice([e for e in pairs if e != opt1])
    else:
        opts[0] = (source, lang_dict[source])

    if target == "xx" or target == "yy":
        opt1 = (source, lang_dict[source]) if source in lang_dict else ("", "")
        opts[1] = random.choice([e for e in pairs if e != opt1])
    else:
        opts[1] = (target, lang_dict[target])
        
    return random.sample(opts, 2)

def get_batch_pairs(batch_size, data, src, tgt, lang_dict):
    (l1, long1), (l2, long2) = sample_langs(src, tgt, lang_dict)
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

    parser = argparse.ArgumentParser(description="Finetuning script for NLLB models.")

    parser.add_argument("--src", type=str, required=True, help="Source language id")
    parser.add_argument("--tgt", type=str, required=True, help="Target language id")
    parser.add_argument("--csv", type=str, required=True, help="CSV containing parallel sentences")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate at end of training")
    parser.add_argument("--model_dir", type=str, help="Directory for storing the trained model")
    parser.add_argument("--nllb_model", type=str, default="600M", choices=['600M', '1.3B', '3.3B'])
    args = parser.parse_args()
    src_lang_code = args.src
    model_save_path = args.model_dir
    csv_file = args.csv

    if not os.path.exists(csv_file):
        print("csv file does not exist")
        exit()
    if os.path.exists(model_save_path):
        print("model directory already exists")
        exit()
          
    model_name = "facebook/nllb-200-distilled-" + size
    date_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S") # mm/dd/YY H:M:S


    lang_dict = SEED_CODES

    trans_df = pd.read_csv(csv_file, sep=",")

    df_train = trans_df[trans_df.split=='train'].copy() 
    df_dev = trans_df[trans_df.split=='dev'].copy()     
    df_test = trans_df[trans_df.split=='test'].copy()   

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.cuda()
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

    
    x, y, train_loss = None, None, None
    x_dev, y_dev, dev_loss = None, None, None
    best_dev_loss = None
    last_best = 0
    patience = 30000
    cleanup()

    for i in tqdm(range(len(train_losses), training_steps)):
        xx, yy, lang1, lang2 = get_batch_pairs(batch_size, df_train, args.src, args.tgt, lang_dict)
        xx_dev, yy_dev, lang1_dev, lang2_dev = get_batch_pairs(batch_size, df_dev, args.src, args.tgt, lang_dict)

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
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            best_dev_loss = dev_loss
            last_best = i
        
        if i - last_best >= patience:
            break

    with open(model_save_path + "/info.txt", 'w') as file:
        file.write(f"Model training started at {date_time}\n")
        file.write(f"pretrained model: {model_name}\n")
        file.write(f"src: {args.src}\n")
        file.write(f"tgt: {args.tgt}\n")
            
    if evaluate:
        evaluate.main(sys.argv[:-1])