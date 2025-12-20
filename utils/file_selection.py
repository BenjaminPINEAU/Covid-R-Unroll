import os 
import re
from collections import defaultdict


def collect_files_by_country(folder, countries):

    pattern = re.compile(r"data_(\w+)_70weeks(\d+)")
    files_by_country = defaultdict(list)

    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            coutry, idx = m.group(1), int(m.group(2))
            if coutry in countries:
                files_by_country[coutry].append((idx, fname))
    for c in countries:
        files_by_country[c].sort(key= lambda x: x[0])
    
    return files_by_country


def build_group_by_pairs(files_by_country, countries):
    
    groups = []
    n_groups = min(len(files_by_country[c]) for c in countries)

    for i in range(n_groups):
        group = [files_by_country[c][i][1] for c in countries]
        groups.append(group)

    return groups

def build_group_by_country(files_by_country, countries):

    groups = []
    n_groups = min(len(files_by_country[c]) for c in countries)

    for c in countries:
        group = [files_by_country[c][i][1] for i in range(n_groups)]
        groups.append(group)
        
    return groups