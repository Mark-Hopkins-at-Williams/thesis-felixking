# thesis-felixking
## Repository for thesis research on the Meta AI NLLB models

Early work for the purpose of getting familiar with these models and finetuning them can be found in code/finetuning.

More recent work on the comparison of encodings of sentences/tokens is in the directory "current." The basic pipeline for running experiments includes the following files:

------------------------------------------------------------------------------------------------------------------------------

## make_dataframe.py
### inputs
1 - number of sentences | 2 - corpus directory | 3 - output stem
### example
python make_dataframe.py 100 /mnt/storage/fking/data/europarl_line_by_line /mnt/storage/fking/data/europarl/europarl_100

this creates a csv and pkl files containing 100 parallel sentences from the data in europarl_line_by_line. The pkl file holds embeddings of the tokens in each sentences as created by both the 600M and 1.3B size 

------------------------------------------------------------------------------------------------------------------------------

## translate.py
### inputs
1 - configuration json file
### example
python translate.py exp_configs/europarl.json

Translates all the sentences in the chosen csv to the chosen language and saves the translations in the same csv convenient for later use in translation_similarity.py.

------------------------------------------------------------------------------------------------------------------------------

## encoding_similarity.py
### inputs
1 - configuration json file
### example
python encoding_similarity.py exp_configs/europarl.json

Uses the stored encodings in the chosen pkl along with faiss to calculate similarity scores for each languagepair based on the average maximum token cosine similarity across all sentences. Makes a heatmap to display similarities and stores the result in chosen experiment directory.

------------------------------------------------------------------------------------------------------------------------------

## translation_similarity.py
### inputs
1 - configuration json file
### example
python encoding_similarity.py exp_configs/europarl.json

Uses the stored translations in the chosen csv along with bleu/chrF++ to calculate similarity scores for each language pair based on the average similarity score across all sentences. Makes a heatmap to display similarities and stores the result in chosen experiment directory.

------------------------------------------------------------------------------------------------------------------------------