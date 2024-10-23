import os
import argparse
import statistics
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from configure import AMERICAS_NLP_CSV, AMERICAS_NLP_LPS
from configure import NLLB_SEED_CSV, NLLB_SEED_LPS
from multilingualdata import MultilingualCorpus
import evaluate

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
    return tokenizer.batch_decode(result, skip_special_tokens=True)


def batched_translate(texts, batch_size=16, **kwargs):
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in tqdm(range(0, len(texts2), batch_size)):
        results.extend(translate(texts2[i: i+batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]


def evaluate_translations(candidate_translations, reference_translations):
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result  = bleu_calc.compute(predictions=candidate_translations, references=reference_translations)
    chrf_result = chrf_calc.compute(predictions=candidate_translations, references=reference_translations)
    return round(bleu_result['score'], 3), round(chrf_result['score'], 3)


def log(src, tgt, bleu_result, chrf_result, model_name, results_path):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")    # mm/dd/YY H:M:S

    sep = " --- "

    output = f"{model_name}{sep}{src}-{tgt}{sep}{date_time}\n{bleu_result}\n{chrf_result}\n\n"

    if os.path.exists(results_path):
        with open(results_path, 'a') as file:
            file.write(output)
    else:
        # File does not exist, create it
        with open(results_path, 'w') as file:
            file.write(output)


def logSet(src, tgt, scores, model_name, results_path):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")    # mm/dd/YY H:M:S

    sep = " --- "

    average_bleu = statistics.mean([b.score for ((s, t), (b, c)) in scores])
    average_chrf = statistics.mean([c.score for ((s, t), (b, c)) in scores])

    output = f"{model_name}{sep}{src}-{tgt}{sep}{date_time}" + ''.join([f"\n\t{t}-{s}:\n\t· {b}\n\t· {c}" for ((t, s), (b, c)) in scores]) + f"\n\taverage bleu: {average_bleu}\n\taveragechrf: {average_chrf}\n\n"

    if os.path.exists(results_path):
        with open(results_path, 'a') as file:

            file.write(output)
    else:
        # File does not exist, create it
        with open(results_path, 'w') as file:

            file.write(output)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation script for NLLB models.")

    # python evaluate.py --data nllb-seed --src lij_Latn --tgt eng_Latn --model_dir /mnt/storage/fking/models/seed_5654 --nllb_model 600M
    parser.add_argument("--data", type=str, required=True, choices=['nllb-seed', 'americas-nlp'], help="Finetuning data")
    parser.add_argument("--src", type=str, required=True, help="Source language id")
    parser.add_argument("--tgt", type=str, required=True, help="Target language id")
    parser.add_argument("--model_dir", type=str, help="Directory for storing the trained model")
    parser.add_argument("--nllb_model", type=str, default="600M", choices=['600M', '1.3B', '3.3B'])
    args = parser.parse_args()

    model_dir = args.model_dir
    model_name = model_dir.split("/")[-1]
    base_model = "facebook/nllb-200-distilled-" + args.nllb_model
    if not os.path.exists(model_dir):
        print(f"model directory doesn't exist: {model_dir}")
        exit()
    
    results_path = f"results.txt"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    #model = AutoModelForSeq2SeqLM.from_pretrained(base_model).cuda() #ONLY FOR TESTING THE BASE MODEL
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).cuda()

    csv_file = NLLB_SEED_CSV if args.data == 'nllb-seed' else AMERICAS_NLP_CSV
    lps = NLLB_SEED_LPS if args.data == 'nllb-seed' else AMERICAS_NLP_LPS
    corpus = MultilingualCorpus(csv_file)

    if args.src == "xx":
        scores = []
        for (s, t) in tqdm(lps):
            print(f"s: {s}, t: {t}")
            dev_bitext = corpus.create_bitext(s, t, 'dev')   
            src_texts, tgt_texts = dev_bitext.lang1_sents, dev_bitext.lang2_sents
            candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=dev_bitext.lang1_code, tgt_lang=dev_bitext.lang2_code)

            bleu_score, chrf_score = evaluate_translations(candidate_translations, tgt_texts)
            scores.append(((s, t), (bleu_score, chrf_score)))
        logSet(args.src, args.tgt, scores, model_name, results_path)
    else:
         
        dev_bitext = corpus.create_bitext(args.src, args.tgt, 'dev')   
        src_texts, tgt_texts = dev_bitext.lang1_sents, dev_bitext.lang2_sents
        candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=dev_bitext.lang1_code, tgt_lang=dev_bitext.lang2_code)

        bleu_score, chrf_score = evaluate_translations(candidate_translations, tgt_texts)
        log(args.src, args.tgt, bleu_score, chrf_score, model_name, results_path)




