import pandas as pd

DATA_PATH = "/mnt/storage/fking/americasnlp2024/ST1_MachineTranslation/data/nahuatl-spanish/"

def get_def_train():
    with open(DATA_PATH + 'train.es', 'r') as file:
        train_es = file.readlines()
    train_es = [line.strip() for line in train_es]
    df_es_train = pd.DataFrame(train_es, columns=['spa'])

    with open(DATA_PATH + 'train.nah', 'r') as file:
        train_nah = file.readlines()
    train_nah = [line.strip() for line in train_nah]
    df_nah_train = pd.DataFrame(train_nah, columns=['nah'])

    return pd.concat([df_es_train, df_nah_train], axis=1)


def get_def_dev():
    with open(DATA_PATH + 'dev.es', 'r') as file:
        dev_es = file.readlines()
    dev_es = [line.strip() for line in dev_es]
    df_es_dev = pd.DataFrame(dev_es, columns=['spa'])

    with open(DATA_PATH + 'dev.nah', 'r') as file:
        dev_nah = file.readlines()
    dev_nah = [line.strip() for line in dev_nah]
    df_nah_dev = pd.DataFrame(dev_nah, columns=['nah'])

    return pd.concat([df_es_dev, df_nah_dev], axis=1)

