#! /usr/bin/python3
# Phillip Richter-Pechanski
# Created on 02.12.2016
# Example:
# fe = FeatureExtractor(samples)
# feature_vectors_as_dicts = fe.extract_baseline_features()
import sys, os, io, re, pprint

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


from collections import OrderedDict
from ml_ner.feature_extraction.word_helpers import word_helper
from ml_ner.feature_extraction.context_helpers import context_helper
from ml_ner.feature_extraction.feature_engeneering_helper import feature_engeneering_helper


class FeatureExtractor:
    '''
    FeatureExtractor class
    '''

    def __init__(self, ne_list, set, filtered = False, verbose = False):
        '''
        Constructor for the FeatureExtractor
        :param ne_list:
        :param set:
        :param verbose:
        '''
        self.ne_list = ne_list
        self.set = set
        self.filtered = filtered
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


    def define_all_features(self):
        '''
        This methods defines the full features
        :return: an ordered dict with the featured as keys
        '''

        # Get the list of not used features
        unused_features = feature_engeneering_helper()
        
        dict_features = OrderedDict()
        '''
        if self.verbose: print("Creating is_np feature")
        dict_features['is_np'] = 0
        if self.verbose: print("Creating is_in_wiki features")
        dict_features['is_in_wiki'] =  0
        if self.verbose: print("Creating is_title features")
        dict_features['is_title'] = 0
        '''
        if self.verbose: print("Creating is_all_caps features")
        dict_features['is_all_caps'] = 0
        '''
        if self.verbose: print("Creating is_name features")
        dict_features['is_name'] = 0
        if self.verbose: print("Creating is_com_name features")
        dict_features['is_com_name'] = 0
        if self.verbose: print("Creating contains_dash features")
        dict_features['contains_dash'] = 0
        '''
        if self.verbose: print("Creating contains_digit features")
        dict_features['contains_digit'] = 0
        
        if self.verbose: print("Creating lemma features")
        lemmas = word_helper(self.set)
        if self.verbose: print("Defining lemma features")
        for lemma in lemmas:
            if lemma.lower().strip() not in unused_features and self.filtered:
                dict_features[lemma.lower().strip()] = 0
            if not self.filtered:
                dict_features[lemma.lower().strip()] = 0
        
        if self.verbose: print("Creating context features")
        contexts = context_helper(self.set)
        if self.verbose: print("Defining context features")
        for context in contexts:
            if context.lower().strip() not in unused_features and self.filtered:
                dict_features[context.lower().strip()] = 0
            if not self.filtered:
                dict_features[context.lower().strip()] = 0
        
        # Define pos-tags
        pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
                    'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                    'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        if self.verbose: print("Creating POS-tag features")
        for pos in pos_tags:
            dict_features[pos] = 0

        # Label
        dict_features['class'] = ""

        if self.verbose: print("Number of features: ", len(dict_features))
        return dict_features

    def extract_all_features(self):
        '''
        This method extracts the feature values per sample
        :return: list of dicts with features and its values extracted per sample
        The following features are extracted:
        @ATTRIBUTE "is_np" REAL
        @ATTRIBUTE "is_in_wiki" REAL
        @ATTRIBUTE "is_title" REAL
        @ATTRIBUTE "is_all_caps" REAL
        @ATTRIBUTE "is_name" REAL
        @ATTRIBUTE "is_com_name" REAL
        #@ATTRIBUTE "contains_dash" REAL
        #@ATTRIBUTE "contains_digit" REAL
        @ATTRIBUTES all lemmas in the corpus REAL
        @ATTRIBUTES all contexts in the corpus REAL
        @ATTRIBUTE class
        returns a list of dicts with the features per sample
        '''
        i = 0

        all_features = self.define_all_features()
        result = []

        # create list of all entries in wikipedia
        '''        
        try:
            wiki_articles = set()
            fname = "../misc/enwiki-latest-all-titles"
            with open(fname, "r") as f:
                for line in f:
                    reg = re.search("\t.*", line)
                    line = reg.group().replace('\t', '').replace('"', "")
                    wiki_articles.add(line.strip().lower())
        except:
            print("Cannot read wikifile")
        '''

        # create list of titles
        try:
            titles_list = set()
            fname = "../misc/title_list"
            with open(fname, "r") as f:
                for line in f:
                    line = line.strip().lower()
                    titles_list.add(line)
        except:
            print("Can not read file with defined titles")

        # create list of commercial names
        try:
            com_list = []
            fname = "../misc/com_list"
            with open(fname, "r") as f:
                for line in f:
                    line = line.strip().lower()
                    com_list.append(line)
        except:
            print("Can not read file with defined commercials")

        # create list of names
        try:
            names_list = set()
            fname = "../misc/names_list"
            with open(fname, "r") as f:
                for line in f:
                    line = line.strip().lower().capitalize()
                    names_list.add(line)
        except:
            print("Can not read file with defined names")

        for sample in self.ne_list:
            i += 1
            label = list(sample)[0]
            if i % 100 == 0 and self.verbose:
                print("Processed ", i, " samples")
            # List of lemmas in sample
            sample_lemmas = [lemma_list[0] for key, value in sample.items() for lemma_list in value if
                             type(lemma_list) == list]
            # List of pos in sample
            sample_pos = [lemma_list[1] for key, value in sample.items() for lemma_list in value if
                          type(lemma_list) == list]

            # Context of current sample
            sample_context = [value[-1] for key, value in sample.items()]

            # Phrase of current sample
            sample_phrase = [value[-2] for key, value in sample.items()]

            # Create dict vector for current sample
            sample_features = all_features.copy()


            # Loop through the features
            for feature in sample_features:
                # Count for each lemma
                for lemma in sample_lemmas:
                    lemma = lemma.lower().strip()
                    feature_l = feature.lower().strip()
                    if lemma == feature_l and lemma != 'class':
                        sample_features[feature] += 1
                
                # Count for each context
                for context in sample_context:
                    context = "_".join(context)
                    context = context.lower().strip()
                    feature_c = feature.lower().strip()
                    if context == feature_c:
                        sample_features[feature] += 1
                
                # Count for each pos-tag
                for pos in sample_pos:
                    if pos == feature:
                        sample_features[pos] += 1
            '''    
            # Check if is NP
            if 'NP' in sample_phrase:
                sample_features['is_np'] = 1
            
            # Check if it is in wiki
            sample_name = "_".join(sample_lemmas)
            sample_name = sample_name.lower()
            if sample_name in wiki_articles:
                sample_features['is_in_wiki'] = 1
            
            # Check if it contains a title
            for lemma in sample_lemmas:
                if lemma.lower() in titles_list:
                    sample_features['is_title'] = 1
            
            # Check if it contains a name
            for lemma in sample_lemmas:
                if lemma.lower() in names_list:
                    sample_features['is_name'] = 1
            
            # Check if it contains a commercial name
            for lemma in sample_lemmas:
                if lemma.lower() in com_list:
                    sample_features['is_com_name'] = 1
            '''
            # Check if one word is all caps
            for lemma in sample_lemmas:
                reg = re.match("[a-zA-Z]", lemma)
                if lemma.isupper() and reg:
                    sample_features['is_all_caps'] = 1
            '''
            # Maybe not helping
            # Check if one word contains a dash
            for lemma in sample_lemmas:
                if "-" in lemma:
                    sample_features['contains_dash'] = 1
            ''' 
            # Check if entity contains digi and/or letterst
            _digits = re.compile('\d')
            for lemma in sample_lemmas:
                if _digits.search(lemma):
                    sample_features['contains_digit'] = 1
                    _letters = re.compile('[a-zA-Z]')
                    if ('-' in lemma or '-' in lemma or ',' in lemma) and not _letters.search(lemma):
                        sample_features['contains_digit'] = 2
                    elif ('-' in lemma or '-' in lemma or ',' in lemma) and _letters.search(lemma):
                        sample_features['contains_digit'] = 3
            
            
            # Set class
            sample_features['class'] = label

            # Add the dict vector of this sample to the result list
            result.append(sample_features)

        return result
