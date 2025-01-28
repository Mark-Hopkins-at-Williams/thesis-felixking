import re
import os
import numpy as np

path = '/mnt/storage/fking/thesis-felixking/experiments/'
exp_dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def calculate_and_write_avgs():
    similarities = {}
    for dir in exp_dirs:

        record_file = os.path.join(dir, 'similarities.txt')
        avg_file = os.path.join(dir, 'avg_similarity.txt')

        with open(record_file, 'r') as file:
            pattern = r'(.+), (.+): (0\.\d+)'

            for line in file:
                match = re.search(pattern, line)
                if match:
                    if match.group(1) not in similarities:
                        similarities[match.group(1)] = []
                    if match.group(2) not in similarities:
                        similarities[match.group(2)] = []
                    similarities[match.group(1)].append(float(match.group(3)))
                    similarities[match.group(2)].append(float(match.group(3)))

        with open(avg_file, 'w') as file:
            file.write('\n'.join([f'{key}: {np.mean(similarities[key])}' for key in similarities]))
    
def calculate_correlation():
    avg_similarities = {}
    avg_file = os.path.join(dir, 'avg_similarity.txt')

    with open(avg_file, 'r') as file:
        pattern = r'(.+) (0\.\d+)'

        for line in file:
            match = re.search(pattern, line)
            if match:
                avg_similarities[match.group(1)].append(float(match.group(2)))


if __name__ == '__main__':
    exit()