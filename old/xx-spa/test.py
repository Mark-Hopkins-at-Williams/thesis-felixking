import random
from americasnlp import SEED_CODES

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


if __name__ == "__main__":
    for i in range(10):
        print(sample_langs("yy", "eng", SEED_CODES))
