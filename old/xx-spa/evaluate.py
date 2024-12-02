import os
import sys
import argparse
import sacrebleu # type: ignore
import pandas as pd
import statistics
from tqdm import tqdm
from datetime import datetime
from transformers import NllbTokenizer
from americasnlp import SEED_CODES
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def translate(
        text, model, tokenizer, src_lang, tgt_lang, 
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


def main(args):

    parser = argparse.ArgumentParser(description="Evaluation script for NLLB models.")

    parser.add_argument("--src", type=str, required=True, help="Source language id")
    parser.add_argument("--tgt", type=str, required=True, help="Target language id")
    parser.add_argument("--csv", type=str, required=True, help="CSV containing parallel sentences")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate at end of training")
    parser.add_argument("--model_dir", type=str, help="Directory for storing the trained model")
    parser.add_argument("--nllb_model", type=str, default="600M", choices=['600M', '1.3B', '3.3B'])
    parser.add_argument("--tag", type=str, choices=['rus-tyv', 'americas', 'seed'])
    args = parser.parse_args()
    model_name = args.model_dir

    csv_file = args.csv
    tgt = args.tgt
    src = args.src
    

    results_path = f"/mnt/storage/fking/thesis-felixking/results/{args.tag}/results.txt"

    model_tok_name = "facebook/nllb-200-distilled-" + args.nllb_model 
    tokenizer = AutoTokenizer.from_pretrained(model_tok_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.nllb_model).cuda()

        



    bleu_calc = sacrebleu.BLEU()
    chrf_calc = sacrebleu.CHRF(word_order=2)  

    #dev set
    trans_df = pd.read_csv(csv_file, sep=",")
    df_dev = trans_df[trans_df.split=='dev'].copy()     


    if src in SEED_CODES:
        translations = batched_translate(df_dev[src].tolist(), model=model, tokenizer=tokenizer, src_lang=src, tgt_lang=tgt)
        bleu_result  = bleu_calc.corpus_score(translations, [df_dev[tgt].tolist()]).score
        chrf_result = chrf_calc.corpus_score(translations, [df_dev[tgt].tolist()]).score
    else: #assume src is not xx or yy
        sources = [e for e in SEED_CODES.keys() if e != tgt]
        bleu_scores = []
        chrf_scores = []

        for i, key in enumerate(sources, start=0):
            translations += batched_translate(df_dev[key].tolist(), src_lang=key, tgt_lang=tgt)
            bleu_scores[i] = bleu_calc(bleu_calc.corpus_score(translations, [df_dev[tgt].tolist()])).score
            chrf_scores[i] = chrf_calc(chrf_calc.corpus_score(translations, [df_dev[tgt].tolist()])).score
        
        bleu_result = statistics.mean(bleu_scores)
        chrf_result = statistics.mean(chrf_scores)


    
    now = datetime.now()
    # mm/dd/YY H:M:S
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")

    sep = " --- "

    output = f"{model_name}{sep}{date_time}\n{bleu_result}\n{chrf_result}\n\n"

    if os.path.exists(results_path):
        with open(results_path, 'a') as file:

            file.write(output)
    else:
        # File does not exist, create it
        with open(results_path, 'w') as file:

            file.write(output)

    
if __name__ == "__main__":
    main(sys.argv)



