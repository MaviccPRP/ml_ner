# This script tests the feature_extrator class

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from ml_ner.homework3.feature_extractor import FeatureExtractor
from ml_ner.homework3.corpusreader import CorpusReader



# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/development/data/english/annotations/nw/wsj")

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()

# Create instance of FeatureEtractor, define features from training set and print verbose
fe = FeatureExtractor(ne, 'train', True)

# Print the extracted features from development set
pprint.pprint(fe.extract_baseline_features())