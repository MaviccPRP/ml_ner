# This script is a helper for the word extraction
# It extracts all lemmas in the corpus and saves lemmas, which occur >=3 times in a list

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint

def feature_engeneering_helper():
    '''
    This class takes a list of engeneered featured, not to be used in the dataset. featured selected by weka feature selection
    '''
    features = []
    try:
        with open("../misc/list_of_unused_features", "r") as f:
            features = f.readlines()
            features = [word.strip() for word in features]
    except:
        print("Cannot open feature file")

    return features