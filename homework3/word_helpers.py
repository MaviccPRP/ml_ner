# This script is a helper for the word extraction
# It extracts all lemmas in the corpus and saves lemmas, which occur >=5 times in a list

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from collections import defaultdict

from ml_ner.homework3.corpusreader import CorpusReader


def word_helper(corpus_set=""):
    '''
    create a list of all used words in the corpus and return them as a list.
    The list contains words which appear at least 3 times in the corpus and do not appear in a stopwordlist
    :param corpus_set:
    :return:
    '''
    stopwords = []
    try:
        with open("english_stopwords", "r") as f:
            stopwords = f.readlines()
            stopwords = [word.strip() for word in stopwords]
    except:
        pass

    corpus_set = "/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/"+corpus_set+"/data/english/annotations/nw/wsj"
    # /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
    # Create an instance of the CorpusReader class
    cr = CorpusReader(corpus_set)

    # Extract the NE and its POS tags
    ne = cr.extract_labeled_named_entities()

    word_list = []
    most_frequent_list = []
    d = defaultdict(int)

    for dictio in ne:
        for k, v in dictio.items():
            for word in v:
                if type(word) == list and word[0] not in stopwords:
                    word_list.append(word[0])

    for word in word_list:
        d[word] += 1

    for key, value in d.items():
        if value >=5:
            most_frequent_list.append(key)

    return sorted(list(set(most_frequent_list)))
