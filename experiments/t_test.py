import numpy as np
import scipy.stats as stats     # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns           # type: ignore
import pandas as pd
import sys
import os
import re

inter = [
    0.7382061595008487,
    0.7054615049135118,
    0.6944264059975034,
    0.7149921428589594,
    0.744381552650815,
    0.7630083929924738,
    0.7485241861570449,
    0.5900211987041292,
    0.6353815964290074,
    0.7400091943286714,
    0.6636073163577488,
    0.7209254446483794,
    0.634763893626985,
    0.6546683424995059,
    0.7036252390770685,
    0.7075407476652236,
    0.7338017054966518,
    0.7207782864570618,
    0.7113079116457984,
    0.6612273653348287,
    0.6978256106376648,
]
intra = [
    0.30893611907958984,
    0.3104860186576843,
    0.3222861886024475,
    0.3129338026046753,
    0.3098888695240021,
    0.3157147765159607,
    0.3215292990207672,
    0.3531643748283386,
    0.3259575068950653,
    0.30185553431510925,
    0.33288049697875977,
    0.30987516045570374,
    0.3293869197368622,
    0.32985660433769226,
    0.30972009897232056,
    0.30988916754722595,
    0.30490806698799133,
    0.3111221194267273,
    0.3183162808418274,
    0.34057119488716125,
    0.3120120167732239,
]

def t_tests(set1, set2):
    assert len(set1) == len(set2)
    t_stat, p_value = stats.ttest_ind(set1, set2)
    print(f"Independent t-test: t={t_stat:.4f}, p={p_value:.4f}")

    # 2. Paired samples t-test (comparing measurements on same subjects)
    t_stat, p_value = stats.ttest_rel(set1, set2)
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

    # 4. Welch's t-test (when equal variances cannot be assumed)
    t_stat, p_value = stats.ttest_ind(set1, set2, equal_var=False)
    print(f"Welch's t-test: t={t_stat:.4f}, p={p_value:.4f}")

if __name__ == '__main__':

    assert len(sys.argv) == 2
    exp_dir = sys.argv[1]

    df = pd.read_csv(os.path.join(exp_dir, 'similarities.csv'))
    interlingual = list(zip(df['language'], df['avg_sim']))

    pattern = r'(.+): (\d+\.\d+)'
    text = ''
    with open(os.path.join(exp_dir, 'intralingual_similarities.txt')) as file:
        text = file.read()

    matches = re.findall(pattern, text)
    intralingual = []
    for match in matches:
        intralingual.append(match[0], match[1])

    