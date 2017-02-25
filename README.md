# Features Evaluation in Named Entity Classification


## Table of contents

- [Synopsis](#synopsis)
- [Contributors](#contributors)
- [Prerequisites](#prerequisites)
- [Folder Structure](#folder-structure)
- [Quickstart](#quickstart)
- [What's included](#whats-included)


## Synopsis

This project focuses on Named Entity Classification. It uses simple syntactic features, frequently used in research, and picks the most significant features. 
For further project details and the used literature see the [final presentation](https://github.com/MaviccPRP/ml_ner/blob/master/reports/presentation_final.pdf) in the documentation/pdf/ folder.

You can find the whole project on GitHub:

https://github.com/MaviccPRP/ml_ner

## Contributors

[M. Huvar](https://github.com/XMadiX)
[Ph. Richter-P.](https://github.com/MaviccPRP)
[S. Safdel](https://github.com/Ssanaz)

## Prerequisites

* Python 3.4+
	* Scikit Learn als Klassifizierer
	* liac-arff
	* matplotlib
* WEKA (to watch the .arff files)

* A valid OntoNotes corpus 2012 in the in the following location: resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/nw/

* To print the ROC curves, you need the ```$DISPLAY``` environment variable being set.

## Folder Structure

*corpus/*
>Contains the corpus reader class

*feature_extraction/*
>Contains the feature extractor class and the arff_scikitdata_creator class for creating arff files and scikit instances. Additionally helpers for the feature extractor class. 

*literature/pdf/*
>Contains the most important papers, used for this project.

*misc/*
>Contains several lists for feature extraction, e.g. wikipedia titles, name lists and official titles list.

*reports/*
>Contains presentations and the final report

*tests/*
>Contains several test scripts for the classes. For example usage, see section Code Example


## Quickstart

Until now you can vary the featuresset and evaluate each featureselection individually, by just editing the extract_features variable in the test scripts.

In the future, we want to implement a module, which is classifiying a predefined list of Named Entities using the command line.

We recommend using a virtual environment for Python 3.

    $ virtualenv -p python3 venv
    $ source venv/bin/activate  
    $ pip install -r requirements.txt  

Congratulations! You are now ready to evaluate features for Named Entity Classification.

Before starting your evaluations, you can define your featureset in each of the following scripts by editing the extract_features list variable.

To evaluate all features (for a full feature description see, final presentation in reports/) printing the results into a ROC curve, type in the following:

```
python tests/test_roc_curve.py
```

To create an .arff file for further analyses in WEKA, type in:

```
python tests/test_arff_creator.py 
```

To get a confusion matrix and a full evaluation report using scikits classification_report, type in:

```
python tests/test_scidata_creator_alternative.py 
```



## What's included

Classes included:

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
└── literature

```


A list of the most important classes used in the project. (*For the full Pythondocs see the .py files*)

* **class ArffAndSciKitDataCreator**  
    Class to create an arff filefrom a given list of features and values created by the corpus_reader and feature_extractor.

* **class FeatureExtractor**  
    Class to extract a list of predefined features.

* **class CorpusReader**  
    Corpus Reader for the Ontonotes 2012 conll corpus

