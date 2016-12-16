# This script tests the arff creator class

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint

from ml_ner.feature_extraction.feature_extractor import FeatureExtractor
from ml_ner.feature_extraction.arff_creator import ArffCreator
from ml_ner.corpus.corpusreader import CorpusReader



# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/wsj")

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()


fe = FeatureExtractor(ne, True)
# Print the first feature vector dict
samples = fe.extract_baseline_features()

arff = ArffCreator(samples)
arff_list  = arff.generate_arff('test.arff')
