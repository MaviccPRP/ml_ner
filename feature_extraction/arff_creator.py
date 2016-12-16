import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import pprint as pp
import arff
from collections import OrderedDict
from ml_ner.feature_extraction.word_helpers import word_helper
from ml_ner.feature_extraction.feature_extractor import FeatureExtractor
from ml_ner.corpus.corpusreader import CorpusReader



class ArffCreator:

    def __init__(self, samples):
        self.samples = samples

    def define_features(self):


        classes = ['PERSON', 'NORP', 'FACILITY', 'ORGANIZATION', 'GPE', 'LOCATION', 'PRODUCT',
                                          'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
                                          'QUANTITY', 'ORDINAL', 'CARDINAL', 'ORG', 'LOC', 'FAC']

        # Create an instance of the CorpusReader class
        cr = CorpusReader(
            "/resources/corpora/multilingual/ontonotes-5.0-conll-2012/conll-2012/v4/data/development/data/english/annotations/nw/wsj")

        # Extract the NE, its POS tags and phrases
        ne = cr.extract_labeled_named_entities()

        fe = FeatureExtractor(ne, True)
        # Print the first feature vector dict
        attributes = fe.define_baseline_features()

        attributes_list = []

        for attribute in attributes:
            if attribute == 'class':
                attributes_list.append((attribute, classes))
            elif "," in attribute or "%" in attribute:
                attributes_list.append(("'" + attribute + "'", 'REAL'))
            else:
                attributes_list.append(('"' + attribute + '"', 'REAL'))

        return attributes_list



    def generate_arff(self, filename):

        samples_dict_to_arff = OrderedDict()
        samples_dict_to_arff['description'] = "Testing arff"
        samples_dict_to_arff['relation'] = "Test"
        samples_dict_to_arff['data'] = []
        samples_dict_to_arff['attributes'] = self.define_features()

        for sample in self.samples:
            temp_sample_list = []
            for key,value in sample.items():
                temp_sample_list.append(value)
            samples_dict_to_arff['data'].append(temp_sample_list)


        save = arff.dumps(samples_dict_to_arff)
        f = open(filename, 'w')
        f.write(save)