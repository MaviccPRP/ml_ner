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
from ml_ner.corpus.corpusreader import CorpusReader


'''
Extract features for the training set
'''

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

'''
Extract features for the test set
'''

# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/test/data/english/annotations/nw/wsj", 'auto')

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()

fe = FeatureExtractor(ne, 'train', False, True)

# Extract features
samples = fe.extract_all_features()

data = ArffAndSciKitDataCreator(samples)

# Classify data with sklearn

X_test, y_test = array(data.createScikitData()[0]), array(data.createScikitData()[1])

'''
print("Classification per sample", flush=True)


for test_sample, entity, f_vector in zip(X_test, ne, samples):
    y_pred_l1_one = OneVsOneClassifier(LinearSVC(random_state=0, dual=False, penalty='l1')).fit(X_train, y_train).predict(test_sample.reshape(1, -1))
    for key, value in entity.items():
        if key != y_pred_l1_one:
            print("Entity: \n"+str(entity)+str(y_pred_l1_one), flush=True)
            for key, value in f_vector.items():
                if not isinstance(value, str) and value > 0:
                    print(str(key) + " : " + str(value), flush=True)
            print("____________________________________________", flush=True)
'''
'''
tuned_parameters = [{'penalty': ['l1'], 'dual': [False], 'multi_class': ['ovr','crammer_singer'],
                     'max_iter': [100,1000,10000], 'C': [1, 10, 100, 1000]},
                    {'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'],'dual': [True], 'multi_class': ['ovr', 'crammer_singer'],
                     'max_iter': [100, 1000, 10000], 'C': [1, 10, 100, 1000]}]

scores = ['accuracy', 'f1_weighted']

labels=['PERSON', 'GPE_NORP', 'ORG', 'DATE', 'PERCENT_CARDINAL_MONEY']


print("Classify LinearSVC")
for score in scores:
    print("Calculating: " + str(score))
    clf = GridSearchCV(LinearSVC(), tuned_parameters, scoring='%s' % score)
    print("Training: " + str(score))
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_test, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_test, y_pred, labels=labels))
    print()

    print("Solely LinearSVC")
    print(accuracy_score(y_test, y_pred_l1))
    print(f1_score(y_test, y_pred_l1, average='weighted'))
    print(confusion_matrix(y_test, y_pred_l1, labels=['PERSON', 'GPE_NORP', 'ORG', 'DATE', 'PERCENT_CARDINAL_MONEY']))
'''
clf = LinearSVC(dual=False, penalty='l1', random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
labels=['PERSON', 'GPE_NORP', 'ORG', 'DATE', 'PERCENT_CARDINAL_MONEY']

print(classification_report(y_test, y_pred, labels=labels))
print(confusion_matrix(y_test, y_pred, labels=labels))
print(accuracy_score(y_test, y_pred))
