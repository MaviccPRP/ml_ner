# Features Evaluation in Named Entity Classification

## Synopsis

This project focuses on Named Entity Classification. It uses simple syntactic features, frequently used in research, and picks the most significant features. 
For further project details and the used literature see the finalpresentation in the documentation/pdf/ folder.

You can find the whole project on GitHub:

https://github.com/MaviccPRP/ml_ner

## Contributors

M. Huvar
https://github.com/XMadiX

Ph. Richter-P.
https://github.com/MaviccPRP

S. Safdel
https://github.com/Ssanaz

## Prerequisites

*Python 3.4+
	*Scikit Learn als Klassifizierer
	*liac-arff
	*matplotlib
5. WEKA (to watch the .arff files)

To print the ROC curves, you need the $DISPLAY environment variable being set.

## Folder Structure

*corpus/*
Contains the corpus reader class

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


## Installation and Examples

We recommend using a virtual environment for Python 3.

    $ virtualenv -p python3 venv
    $ source venv/bin/activate  
    $ pip install -r requirements.txt  

Congratulations! You are now ready to classify sarcasm.

Before starting your evaluations, you can define your featureset in each of the following scripts by editing the extract_features list.

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



## Classes

A list of the classes used in the project. For the full Pythondocs see the .py files.
```
class ArffAndSciKitDataCreator:
    '''
    Class to create an arff filefrom a given list of features and values created by the corpus_reader and feature_extractor.
    '''

class FeatureExtractor:
    '''
    Class to extract a list of predefined features.
    '''

class CorpusReader:
    '''
    Corpus Reader for the Ontonotes 2012 conll corpus
    '''
```

