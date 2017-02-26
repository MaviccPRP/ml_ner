# Features Evaluation in Named Entity Classification


## Table of contents

* [Synopsis](#synopsis)
* [Contributors](#contributors)
* [Prerequisites](#prerequisites)
* [Quickstart](#quickstart)
* [What's included](#whats-included)
  * [Folder descriptions](#folder-descriptions)
* [Classes](#classes)


## Synopsis

This module focuses on Named Entity Classification. You can evaluate a specific set of features, which you can define by just editing a featureset variable, and get evaluation results as a ROC-curve, a confusion matrix and a more comples classification report.
The default setup uses the OntoNotes 2012 corpus in CoNLL format and eleven predefined syntactic features on word and sentence level.
For further project details, predefined features, evaluation results and the used literature see the [final presentation](https://github.com/MaviccPRP/ml_ner/blob/master/reports/presentation_final.pdf) in the reports/ folder.

You can find the whole project on GitHub:

https://github.com/MaviccPRP/ml_ner

## Contributors

* [M. Huvar](https://github.com/XMadiX)
* [Ph. Richter-P.](https://github.com/MaviccPRP)
* [S. Safdel](https://github.com/Ssanaz)

## Prerequisites

* Python 3.4+
	* Scikit Learn als Klassifizierer
	* liac-arff
	* matplotlib
* WEKA (to watch the .arff files)

* A valid OntoNotes corpus 2012 in the in the following location: /resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/

* To print the ROC curves, you need the ```$DISPLAY``` environment variable being set.


## Quickstart

Until now you can vary the featuresset and evaluate each featureselection individually, by just editing the extract_features variable in the test scripts.

In the future, we want to implement a module, which is classifiying a predefined list of Named Entities using the command line.

We recommend using a virtual environment for Python 3.

    $ virtualenv -p python3 venv
    $ source venv/bin/activate  
    $ pip install -r requirements.txt  

Congratulations! You are now ready to evaluate features for Named Entity Classification.

Before starting your evaluations, you can define your featureset in each of the following scripts by editing the extract_features list variable.

**To evaluate all features (for a full feature description see, final presentation in reports/) printing the results into a ROC curve, type in the following:**

```
python tests/test_roc_curve.py
```

Example output ROC curve:

![alt tag](https://github.com/MaviccPRP/ml_ner/blob/master/reports/roc_curve.png)

To get known how an optimal ROC curve needs to look like, have a look at the [Wikipedia article](https://de.wikipedia.org/wiki/Receiver_Operating_Characteristic).

**To create an .arff file for further analyses in WEKA, type in:**

```
python tests/test_arff_creator.py 
```

**To get a confusion matrix and a full evaluation using scikits classification_report, type in:**

```
python tests/test_scidata_creator_alternative.py 
```
Example output of a confusion matrix and a classification report:

```
[[348  27  38   0   0]
 [ 29 549  10   0   0]
 [ 64  52 739   4   0]
 [  0   4   0 583  14]
 [  0   1   0   2 526]]
                        precision    recall  f1-score   support

                PERSON       0.79      0.84      0.81       413
              GPE_NORP       0.87      0.93      0.90       588
                   ORG       0.94      0.86      0.90       859
                  DATE       0.99      0.97      0.98       601
PERCENT_CARDINAL_MONEY       0.97      0.99      0.98       529

           avg / total       0.92      0.92      0.92      2990

```



## What's included

Folder structure and classes included:

```
ml_ner/
├── corpus/
│   └── corpusreader.py
│ 
├── feature_extrator/
│   ├── arff_scikitdata_creator.py
│   ├── context_helpers.py 
│   ├── feature_engeneering_helper.py
│   ├── word_helpers.py
│   └── feature_extractor.py
├── tests/
│   └── evaluation
├── reports 
├── docs
└── literature

```
### Folder descriptions

corpus/
>Contains the corpus reader class

feature_extraction/
>Contains the feature extractor class and the arff_scikitdata_creator class for creating arff files and scikit instances. Additionally helpers for the feature extractor class. 

literature/pdf/
>Contains the most important papers, used for this project.

misc/
>Contains several lists for feature extraction, e.g. wikipedia titles, name lists and official titles list.

reports/
>Contains presentations and the final report

docs/
>The python docs for the classes

tests/
>Contains several test scripts for the classes. For example usage, see section Code Example

## Classes

A list of the most important classes used in the project. (*For the full Pythondocs see in [docs/ml_ner.html](docs/ml_ner.html))

* **class ArffAndSciKitDataCreator**  
    Class to create an arff filefrom a given list of features and values created by the corpus_reader and feature_extractor.

* **class FeatureExtractor**  
    Class to extract a list of predefined features.

* **class CorpusReader**  
    Corpus Reader for the Ontonotes 2012 conll corpus

