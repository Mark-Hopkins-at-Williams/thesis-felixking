import os
import re
import sys

def get_intra_sims(path):
    
    text = ''
    with open(os.path.join(path, 'intralingual_similarities.txt')) as file:
        text = file.read().strip()

    pattern = r'\n?(\D+): (\d+\.\d+)'

    matches = re.finditer(pattern, text)
    pairs = {}
    for match in matches:
        pairs[match.group(1)] = float(match.group(2))

    return pairs

if __name__ == '__main__':
        

    exp_name = ''
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    else:
        print('usage: python rv1_plot.py <exp_name>')
        exit()

    spurious = f'scrambled_{exp_name}'

    if not os.path.isdir(spurious) or not os.path.isdir(exp_name):
        print('path doesn\'t exist')
        exit()


    intra = get_intra_sims('')
    print()