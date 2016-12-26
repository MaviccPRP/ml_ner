# This script tests the scikitdata creator class
# And does a classification on training and testdata via scikit learn

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from numpy import array
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ml_ner.feature_extraction.feature_extractor import FeatureExtractor
from ml_ner.feature_extraction.arff_scikitdata_creator import ArffAndSciKitDataCreator
from ml_ner.corpus.corpusreader import CorpusReader



# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/wsj")

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()


fe = FeatureExtractor(ne, 'train', False, True)

# Extract features
samples = fe.extract_all_features()

data = ArffAndSciKitDataCreator(samples)

# Classify data with sklearn

X_train, y_train = array(data.createScikitData()[0]), array(data.createScikitData()[1])

# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/test/data/english/annotations/nw/wsj")

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()

fe = FeatureExtractor(ne, 'train', False, True)

# Extract features
samples = fe.extract_all_features()

data = ArffAndSciKitDataCreator(samples)

# Classify data with sklearn

X_test, y_test = array(data.createScikitData()[0]), array(data.createScikitData()[1])

print("Classify one-vs-one")
y_pred_l1_one = OneVsOneClassifier(LinearSVC(random_state=0, dual=False, penalty='l1')).fit(X_train, y_train).predict(X_test)

print(accuracy_score(y_test, y_pred_l1_one))