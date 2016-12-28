# This script tests the arff creator class

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint

from ml_ner.feature_extraction.feature_extractor import FeatureExtractor
from ml_ner.feature_extraction.arff_scikitdata_creator import ArffAndSciKitDataCreator
from ml_ner.corpus.corpusreader import CorpusReader



# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/wsj")

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()


fe = FeatureExtractor(ne, 'train', False, True)

# Define features for the ARFF file
sample_features = fe.define_all_features()

# Extract features
samples = fe.extract_all_features()

data = ArffAndSciKitDataCreator(samples, sample_features)
arff_list  = data.generate_arff('train.arff')