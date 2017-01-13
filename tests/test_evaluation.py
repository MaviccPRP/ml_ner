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

from sklearn.preprocessing import label_binarize
from sklearn.metrics import hinge_loss
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle

from ml_ner.feature_extraction.feature_extractor import FeatureExtractor
from ml_ner.feature_extraction.arff_scikitdata_creator import ArffAndSciKitDataCreator
from ml_ner.corpus.corpusreader import CorpusReader

from itertools import combinations

import numpy as np

extract_features = ['is_np','in_wiki','is_title','is_all_caps','is_name','is_com_name','contains_dash','contains_digit','lemma','context','pos']



for L in range(0, len(extract_features)+1):
    for subset in itertools.combinations(extract_features, L):
        '''
        Extract features for the training set
        '''

        # /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
        # Create an instance of the CorpusReader class
        cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/wsj", "auto")

        # Extract the NE, its POS tags and phrases
        ne = cr.extract_labeled_named_entities()

        fe = FeatureExtractor(ne, 'train', extract_features, False, True)

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
        cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/test/data/english/annotations/nw/wsj", 'auto')

        # Extract the NE, its POS tags and phrases
        ne = cr.extract_labeled_named_entities()

        fe = FeatureExtractor(ne, 'train', extract_features,False, True)

        # Extract features
        samples = fe.extract_all_features()

        data = ArffAndSciKitDataCreator(samples)

        # Classify data with sklearn

        X_test, y_test = array(data.createScikitData()[0]), array(data.createScikitData()[1])


        labels=['PERSON', 'GPE_NORP', 'ORG', 'DATE', 'PERCENT_CARDINAL_MONEY']
        classifier = OneVsRestClassifier(LinearSVC())
        y_pred = classifier.fit(X_train, y_train).predict(X_test)
        print(classification_report(y_test, y_pred, target_names=labels, digits=4))


        # Binarize the output
        y_test = label_binarize(y_test, classes=labels)
        n_classes = y_test.shape[1]


        y_train = label_binarize(y_train, classes=labels)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(LinearSVC())
        y_pred = classifier.fit(X_train, y_train).predict(X_test)
        print(subset)
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("F1-Score: ", f1_score(y_test, y_pred, average='weighted'))
        print("Confusion Matrix: ")
        print(confusion_matrix(y_test, y_pred, labels=['PERSON', 'GPE_NORP', 'ORG', 'DATE', 'PERCENT_CARDINAL_MONEY']))
        print("Classification report: ")
        print(classification_report(y_test, y_pred, labels=labels))