# This script is a helper for the word extraction
# It extracts all lemmas in the corpus and saves lemmas, which occur >=3 times in a list

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from collections import Counter


def context_helper(corpus_set=""):
    '''

    '''
    contexts = []
    try:
        with open("../misc/context_list", "r") as f:
            contexts = f.readlines()
            contexts = [word.strip() for word in contexts]
    except:
        print("Cannot open context file")

    counts = Counter(contexts)

    most_frequent_contexts = []

    for key, value in counts.items():
        if value > 10:
            most_frequent_contexts.append(key)

    return sorted(most_frequent_contexts)