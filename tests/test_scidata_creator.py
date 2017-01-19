# This script tests the scikitdata creator class
# And does a classification on training and testdata via scikit learn

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from numpy import array
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from ml_ner.feature_extraction.feature_extractor import FeatureExtractor
from ml_ner.feature_extraction.arff_scikitdata_creator import ArffAndSciKitDataCreator
from ml_ner.corpus.corpusreader_allclasses import CorpusReader


'''
Extract features for the training set
'''

# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/wsj")

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()

extract_features = ['is_np', 'is_in_wiki', 'is_all_caps', 'contains_digit', 'lemma', 'context', 'pos']

fe = FeatureExtractor(ne, 'train', extract_features,False, True)

# Extract features
samples = fe.extract_all_features()

data = ArffAndSciKitDataCreator(samples)

# Classify data with sklearn

X_train, y_train = array(data.createScikitData()[0]), array(data.createScikitData()[1])

'''
       Extract features for the test set
       '''

# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader(
    "/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/test/data/english/annotations/nw/wsj",
    'auto')

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()

fe = FeatureExtractor(ne, 'train', extract_features, False, True)

# Extract features
samples = fe.extract_all_features()

data = ArffAndSciKitDataCreator(samples)

# Classify data with sklearn

X_test, y_test = array(data.createScikitData()[0]), array(data.createScikitData()[1])

classifier = OneVsRestClassifier(LinearSVC())
y_pred = classifier.fit(X_train, y_train).predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred, average='weighted'))