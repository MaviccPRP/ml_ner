#! /usr/bin/python3
# Phillip Richter-Pechanski
# Created on 28.12.2016
# Example:
# fe = FeatureExtractor(ne, 'train', False, True)
# feature_vectors_as_dicts = fe.extract_baseline_features()

import sys, os, io, re, pprint

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


from collections import OrderedDict
from ml_ner.homework3.word_helpers import word_helper

class FeatureExtractor:
    '''
    FeatureExtractor class
    '''

    def __init__(self, ne_list, set, verbose = False):
        '''
        Constructor for the FeatureExtractor
        :param ne_list:
        :param set:
        :param verbose:
        '''
        self.ne_list = ne_list
        self.set = set
        self.verbose = verbose

    def define_baseline_features(self):
        '''
        This methods defines the baseline features
        This methods defines the baseline features
        :return: an ordered dict with the featured as keys
        '''
        dict_features = OrderedDict()
        # Extract lemmas
        lemmas = word_helper(self.set)
        if self.verbose: print("Defining lemma features")
        for lemma in lemmas:
            dict_features[lemma.lower().strip()] = 0

        # Label
        dict_features['class'] = ""

        if self.verbose: print("Number of features: ", len(dict_features))
        return dict_features


    def extract_baseline_features(self):
        '''
        This method extracts the feature values per sample
        :return: list of dicts with features and its values extracted per sample
        '''
        i = 0
        # Extracts: lemma

        baseline_features = self.define_baseline_features()
        result = []

        for sample in self.ne_list:
            i += 1
            label = list(sample)[0]
            if i % 100 == 0 and self.verbose:
                print("Processed ", i, " samples")
            # List of lemmas in sample
            sample_lemmas = [lemma_list[0] for key, value in sample.items() for lemma_list in value if type(lemma_list) == list]
            # List of pos in sample
            #sample_pos = [lemma_list[1] for key, value in sample.items() for lemma_list in value if type(lemma_list) == list]

            #Create dict vector for current sample
            sample_features = baseline_features.copy()
            # Loop through the features
            for feature in sample_features:
                # Count for each lemma
                for lemma in sample_lemmas:
                    lemma = lemma.lower().strip()
                    feature_l = feature.lower().strip()
                    if lemma == feature_l and lemma != 'class':
                        sample_features[feature] += 1


                # Set class
                sample_features['class'] = label

            # Add the dcit vector of this sample to the result list
            result.append(sample_features)
        return result
