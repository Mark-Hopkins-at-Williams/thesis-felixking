from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

    print(tokens.attention_mask)

    

if __name__ == '__main__':
    base_model = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    print(model)

    sents = ['it rained very hard today', 'the stock market fell one million points today and it rained cats.']

    see_tokens(sents, 'eng_Latn')
    print()