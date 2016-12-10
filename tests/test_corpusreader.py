# This script tests the corpusreader
# Useage: python test_corpusreader.py > test_corpusreader_output.txt
# The results will be printed to test_corpusreader_output.txt

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from ml_ner.corpus.corpusreader import CorpusReader


# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/wsj")

# Extract the NE and its POS tags
ne = cr.extract_labeled_named_entities()

# Pretty print the output
pprint.pprint(ne)



