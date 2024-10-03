import pandas as pd
import os

# This program makes csv files containing all the training and dev data for each of the language
# pairs in the americasnlp dataset. each csv has the headers 
#               [src_lang_code], spa, split

# where split is either "train" or "dev"

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

for subdir, dirs, files in os.walk(parent_path):
    if subdir == parent_path:
        for dir_name in dirs:
            dir_path = os.path.join(subdir, dir_name)
            language = dir_path.split("/")[-1].split("-")[0] #identify the american language

            train = get_def_train(dir_path, codes[language])
            dev = get_def_dev(dir_path, codes[language])

            full = pd.concat([train, dev])

            full.to_csv(dir_path + "/all.csv", index = False)



