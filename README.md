## Synopsis

This project focuses on Named Entity Classification. It uses simple syntactic features, frequently used in research (Toral, Munoz, 2006; Kazama, Torisawa, 2007; Ratinov, Roth 2009), and evaluates the most significant features. 

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

1. Python 3.4+
2. Scikit Learn als Klassifizierer
3. liac-arff
4. matplotlib

To print the ROC curves, you need the $DISPLAY environment variable being set.

## Folder Structure

*corpus/*
Contains the corpus reader class

*feature_extraction/*
Contains the feature extractor class and the arff_scikitdata_creator class for creating arff files and scikit instances. Additionally helpers for the feature extractor class. 

*literature/pdf/*
Contains the most important papers, used for this paper.

*misc/*
Contains several lists for feature extraction, e.g. wikipedia titles, name lists and official titles list.

*reports/*
Contains presentations and the final report

*tests/*
Contains several test scripts for the classes. For example usage, see section Code Example


## Examples

To evaluate all features (for a full feature description see, final_presentation in report/), and write the results into a ROC curve, type in the following:

```
python tests/test_roc_curve.py
```

To create an .arff file for further analyses in WEKA, use:

```
python tests/test_arff_creator.py 
```



## Classes

A list of the classes used in the project. For the full Pythondocs see the .py files.

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


