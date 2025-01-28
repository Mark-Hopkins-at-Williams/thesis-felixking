import os
import re

def removeTag(sent):
    # get rid of (EN) type tags at the beginning of sentences, for example
    match = re.search(r'(\w*\([A-Z]+\)\w*)(.+)', sent) 
    if match:
        sent = match.group(2)
    return sent

class LangPair():

    def __init__(self, filename):
        self.lang1, self.lang2 = filename.split('.')[1].split('-')
        basename, ext = os.path.splitext(filename)

        print(self.lang1, self.lang2)

        eng_file = f'{basename}.{self.lang2}'
        other = f'{basename}.{self.lang1}'

        self.sentPairs = self.read(eng_file, other)


    def read(self, eng, other):
        eng_sents = []
        sents = {}
        with open(eng, 'r') as reader:
            for line in reader:
                eng_sents.append(removeTag(line.strip()))

        i = 0
        with open(other, 'r') as reader:
            for line in reader:
                if len(line) > 5:
                    sents[eng_sents[i]] = removeTag(line.strip())
                i += 1

        print(len(sents))
        return sents
    
    def filter(self, eng_sents):
        self.sentPairs = {k: self.sentPairs[k] for k in eng_sents}

    def writeOther(self):
        keys = sorted(self.sentPairs.keys())

        with open(f'/mnt/storage/fking/data/europarl/filtered/{self.lang1}_sents.txt', 'w') as file:
            for key in keys:
                file.write(self.sentPairs[key] + "\n")

    def writeEn(self, sents):

        with open(f'/mnt/storage/fking/data/europarl/filtered/en_sents.txt', 'w') as file:
            for sent in sorted(sents):
                file.write(sent + "\n")

    
if __name__ == '__main__':

    unfiltered_path = '/mnt/storage/fking/data/europarl/unfiltered'
    filtered_path = '/mnt/storage/fking/data/europarl/filtered'

    lps = []
    for file in os.listdir(unfiltered_path):
        name, ext = os.path.splitext(file)

        if ext == '.en':
            lps.append(LangPair(os.path.join(unfiltered_path, file)))

    eng_sents = set(lps[0].sentPairs.keys())
    for lp in lps[1:]:
        eng_sents &= set(lp.sentPairs.keys())
    eng_sents = sorted([sent for sent in eng_sents if len(sent) > 10])

    print('len sorted eng', len(eng_sents))

    for lp in lps:
        lp.filter(eng_sents)
        lp.writeOther()
    
    lps[0].writeEn(eng_sents)