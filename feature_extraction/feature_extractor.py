#! /usr/bin/python3
# Phillip Richter-Pechanski
# Created on 02.12.2016
# Example:
# fe = FeatureExtractor(samples)
# feature_vectors_as_dicts = fe.extract_baseline_features()
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


from collections import OrderedDict
from ml_ner.feature_extraction.word_helpers import word_helper


class FeatureExtractor:
    '''
    FeatureExtractor class
    '''

    def __init__(self, ne_list, verbose = False):
        '''
        Constructor for the FeatureExtractor
        :param ne_list:
        :param verbose:
        '''
        self.ne_list = ne_list
        self.verbose = verbose

    def define_baseline_features(self):
        '''
        This methods defines the baseline features
        This methods defines the baseline features
        :return: an ordered dict with the featured as keys
        '''
        dict_features = OrderedDict()
        # Extract lemmas
        lemmas = word_helper()
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
        # Extracts: lemma, pos tags

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
            sample_pos = [lemma_list[1] for key, value in sample.items() for lemma_list in value if type(lemma_list) == list]

            #Create dict vector for current sample
            sample_features = baseline_features.copy()

            # Loop through the features
            for feature in sample_features:

                # Count for each lemma
                for lemma in sample_lemmas:
                    lemma = lemma.lower().strip()
                    feature_l = feature.lower().strip()
                    if lemma == feature_l:
                        sample_features[feature] += 1

                # Count for each pos-tag
                for pos in sample_pos:
                    if pos == feature:
                        sample_features[pos] += 1

                # Set class
                sample_features['class'] = label

            # Add the dcit vector of this sample to the result list
            result.append(sample_features)
        return result


    def define_all_features(self):
        '''
        This methods defines the full features
        :return: an ordered dict with the featured as keys
        '''
        dict_features = OrderedDict()
        if self.verbose: print("is_np feature")
        dict_features['is_np'] = False
        if self.verbose: print("Creating is_in_wiki features")
        dict_features['is_in_wiki'] =  False
        if self.verbose: print("Creating is_not_in_wordnet features")
        dict_features['is_not_in_wordnet'] = False
        if self.verbose: print("Creating is_title features")
        dict_features['is_title'] = False
        if self.verbose: print("Creating is_all_caps features")
        dict_features['is_all_caps'] = False
        if self.verbose: print("Creating bigram_context features")
        dict_features['bigram_context'] = ""
        if self.verbose: print("Creating lemma features")
        baseline_features = self.define_baseline_features()
        for feature in baseline_features:
            dict_features[feature] = 0

        # Define pos-tags
        pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
                    'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                    'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        if self.verbose: print("Creating POS-tag features")
        for pos in pos_tags:
            dict_features[pos] = 0

        print(dict_features)

