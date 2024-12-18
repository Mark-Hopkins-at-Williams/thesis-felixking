import pandas as pd
import os

# This program makes csv files containing all the training and dev data for each of the language
# pairs in the americasnlp dataset. each csv has the headers 
#               [src_lang_code], spa, split

# where split is either "train" or "dev"

AMERICASNLP_CODES = { # these are just the codes that americasnlp uses for their files
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

SEED_CODES = {
    "ace": "ace_Arab",
    "ary": "ary_Arab",
    "arz": "arz_Arab",
    "bam": "bam_Latn",
    "ban": "ban_Latn",
    "bho": "bho_Deva",
    "bjn": "bjn_Arab",
    "bug": "bug_Latn",
    "crh": "crh_Latn",
    "dik": "dik_Latn",
    "dzo": "dzo_Tibt",
    "eng": "eng_Latn",
    "fur": "fur_Latn",
    "fuv": "fuv_Latn",
    "gug": "gug_Latn",
    "hne": "hne_Deva",
    "kas": "kas_Arab",
    "knc": "knc_Arab",
    "lij": "lij_Latn",
    "lim": "lim_Latn",
    "lmo": "lmo_Latn",
    "ltg": "ltg_Latn",
    "mag": "mag_Deva",
    "mni": "mni_Beng",
    "mri": "mri_Latn",
    "nqo": "nqo_Nkoo",
    "nus": "nus_Latn",
    "pbt": "pbt_Arab",
    "prs": "prs_Arab",
    "scn": "scn_Latn",
    "shn": "shn_Mymr",
    "srd": "srd_Latn",
    "szl": "szl_Latn",
    "taq": "taq_Tfng",
    "vec": "vec_Latn",
    "zgh": "zgh_Tfng",
}

parent_path = "/mnt/storage/fking/americasnlp2024/ST1_MachineTranslation/data/"
            

def get_def_train(path, code):
    with open(path + '/train.es', 'r') as file:
        train_es = file.readlines()
    train_es = [line.strip() for line in train_es]
    df_tgt_train = pd.DataFrame(train_es, columns=['spa'])

    with open(path + '/train.' + code, 'r') as file:
        train_src = file.readlines()
    train_src = [line.strip() for line in train_src]
    df_src_train = pd.DataFrame(train_src, columns=[code])

    result = pd.concat([df_src_train, df_tgt_train], axis=1)
    result["split"] = "train"
    return result

def get_def_dev(path, code):
    with open(path + '/dev.es', 'r') as file:
        train_es = file.readlines()
    train_es = [line.strip() for line in train_es]
    df_tgt_train = pd.DataFrame(train_es, columns=['spa'])

    with open(path + '/dev.' + code, 'r') as file:
        train_src = file.readlines()
    train_src = [line.strip() for line in train_src]
    df_src_train = pd.DataFrame(train_src, columns=[code])

    result = pd.concat([df_src_train, df_tgt_train], axis=1)
    result["split"] = "dev"

    return result

if __name__ == "__main__":
    for subdir, dirs, files in os.walk(parent_path):
        if subdir == parent_path:
            for dir_name in dirs:
                dir_path = os.path.join(subdir, dir_name)
                language = dir_path.split("/")[-1].split("-")[0] #identify the american language

                train = get_def_train(dir_path, AMERICASNLP_CODES[language])
                dev = get_def_dev(dir_path, AMERICASNLP_CODES[language])

                full = pd.concat([train, dev])

                full.to_csv(dir_path + "/all.csv", index = False)



