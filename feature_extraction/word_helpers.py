# This script is a helper for the word extraction
# It extracts all lemmas in the corpus and saves lemmas, which occur >=3 times in a list

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from collections import defaultdict

from ml_ner.corpus.corpusreader import CorpusReader


def word_helper():
    # /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
    # Create an instance of the CorpusReader class
    cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/development/data/english/annotations/nw/wsj")

    # Extract the NE and its POS tags
    ne = cr.extract_labeled_named_entities()

    word_list = []
    most_frequent_list = []
    d = defaultdict(int)

    for dictio in ne:
        for k, v in dictio.items():
            for word in v:
                if type(word) == list:
                    word_list.append(word[0])

    for word in word_list:
        d[word] += 1

    for key, value in d.items():
        if value >=3:
            most_frequent_list.append(key)


    return sorted(list(set(most_frequent_list)))
