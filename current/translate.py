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
    return tokenizer.batch_decode(result, skip_special_tokens=True)


def batched_translate(texts, batch_size=16, **kwargs):
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in range(0, len(texts2), batch_size):
        results.extend(translate(texts2[i: i+batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]


def evaluate_translations(candidate_translations, reference_translations):
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result  = bleu_calc.compute(predictions=candidate_translations, references=reference_translations)
    chrf_result = chrf_calc.compute(predictions=candidate_translations, references=reference_translations)
    bleu_score = round(bleu_result['score'], 3)
    chrf_score = round(chrf_result['score'], 3)
    print(f"bleu: {bleu_score}\nchrf: {chrf_score}")

    return bleu_score, chrf_score

if __name__ == "__main__":

    config_file = sys.argv[1] 
    print(config_file)
    with open(config_file) as reader:
        config = json.load(reader)

    exp_dir = config['experiment_directory']
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    src = config['source']
    tgt = config['target']
    languages = config['languages']
    csv_file = config['parallel_corpus_csv']
    saving_translations = bool(config['save'])

    for model_size in ['600M', '1.3B']:
        save_dir = os.path.join(exp_dir, model_size)
        base_model = f"facebook/nllb-200-distilled-{model_size}"
        save_path = os.path.join(save_dir, f'{src}-{tgt}_scores.csv')

        print('loading model...')
        tokenizer = AutoTokenizer.from_pretrained(base_model, clean_up_tokenization_spaces=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model).cuda()

        corpus = MultilingualCorpus(csv_file)

        pairs = []
        if src == 'all' and tgt == 'all':
            pairs = [(s, t) for s in languages for t in languages]
        elif src == 'all':
            pairs = [(s, tgt) for s in languages if s != tgt]
        elif tgt == 'all':
            pairs = [(src, t) for t in languages if t != src]
        else:
            pairs = [(src, tgt)]

        scores = pd.DataFrame(columns=['source', 'target', 'bleu', 'chrf++'])
            
        if saving_translations:
            save_col = f"{model_size}_{tgt}_translations"
            df = pd.read_csv(csv_file)
            df[save_col] = ''

        for (s, t) in tqdm(pairs):
            if s == t:
                scores.loc[len(scores)] = {'source': s, 'target': t, 'bleu': 100.0, 'chrf++': 100.0}
                continue
            dev_bitext = corpus.create_bitext(s, t, 'train')   
            src_texts, tgt_texts = dev_bitext.lang1_sents, dev_bitext.lang2_sents

            print(f"translating {s} to {t}")
            candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=dev_bitext.lang1_code, tgt_lang=dev_bitext.lang2_code)
            if saving_translations:
                lang, script = s.split('_')
                mask = ((df['language'] == lang) & (df['script'] == script) & (df['split'] == 'train'))
                df.loc[mask, save_col] = candidate_translations

            bleu_score, chrf_score = evaluate_translations(candidate_translations, tgt_texts)
            scores.loc[len(scores)] = {'source': s, 'target': t, 'bleu': bleu_score, 'chrf++': chrf_score}

        print('saving results...')
        if saving_translations:
            df.to_csv(csv_file)
        scores.to_csv(save_path, index=False)
    shutil.copy(config_file, os.path.join(exp_dir, os.path.basename(config_file)))

