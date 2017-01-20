# This script tests the scikitdata creator class
# And does a classification on training and testdata via scikit learn

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from numpy import array
from sklearn.svm import LinearSVC, SVC
from sklearn import linear_model
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

import numpy as np


'''
Extract features for the training set
'''

# /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/
# Create an instance of the CorpusReader class
cr = CorpusReader("/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/wsj", "auto")

# Extract the NE, its POS tags and phrases
ne = cr.extract_labeled_named_entities()

#extract_features = ['pos', 'lemma', 'context', 'is_all_caps', 'contains_digit']
extract_features = ['is_np', 'is_in_wiki', 'is_all_caps', 'contains_digit', 'lemma', 'context', 'pos']
#extract_features = ['is_np', 'is_all_caps', 'pos', 'contains_digit']
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



# Binarize the output
y_test = label_binarize(y_test, classes=labels)
n_classes = y_test.shape[1]


y_train = label_binarize(y_train, classes=labels)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(LinearSVC())
y_score = classifier.fit(X_train, y_train).decision_function(X_test)



# Compute ROC curve and ROC area for each class
lw = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
threshold = dict()
for i in range(n_classes):
    fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print("++++++++++++++++++++++++++++++++")
    print(y_score[:, i])
    #print(y_test[:, i])
    #print(tpr[i])
    #print(fpr[i])
    #print(threshold[i])
    #print("++++++++++++++++++++++++++++++++")

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
#plt.figure()
#plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)

#plt.plot(fpr["macro"], tpr["macro"],
#         label='macro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)

colors = cycle(['red', 'green', 'blue', 'yellow', 'orange'])
for i, color, label in zip(range(n_classes), colors, labels):
    plt.plot(fpr[i], tpr[i], color=color, lw=4,
             label='ROC curve of class '+label+
             ''.format(i, roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-Kurve für jede Klasse')
plt.rc('legend',fontsize='medium') # using a named size
plt.legend(loc="lower right")
fig = plt.gcf()
plt.show()
fig.savefig("roc_curve.png")
