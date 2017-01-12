# This script tests the feature_extrator class

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from ml_ner.feature_extraction.feature_extractor import FeatureExtractor
from ml_ner.corpus.corpusreader import CorpusReader



# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/wsj")

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()

# Which features to extract?
extract_features = ['lemma', 'context', 'pos', 'is_all_caps']

fe = FeatureExtractor(ne_list=ne, set='train', features=extract_features, filtered=True, verbose=True)
# Print the first feater vector dict

result = fe.extract_all_features()