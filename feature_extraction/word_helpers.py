# This script is a helper for the word extractor

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from ml_ner.corpus.corpusreader import CorpusReader

def word_helper():
    # /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
    # Create an instance of the CorpusReader class
    cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/")

    # Extract the NE and its POS tags
    ne = cr.extract_labeled_named_entities()

    word_list = []

    for dictio in ne:
        for k, v in dictio.items():
            for word in v:
                if type(word) == list:
                    word_list.append(word[0])

    return word_list
