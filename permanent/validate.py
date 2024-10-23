import sacrebleu
from tqdm import tqdm


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


def evaluate(candidate_translations, reference_translations):
    bleu_calc = sacrebleu.BLEU()
    chrf_calc = sacrebleu.CHRF( 
        word_order=0, 
        char_order=6, 
        lowercase=False, 
        whitespace=False
    )
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result  = bleu_calc.corpus_score(candidate_translations, reference_translations).score
    chrf_result = chrf_calc.corpus_score(candidate_translations, reference_translations).score
    return bleu_result, chrf_result

    
  