import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import arff, pprint
from collections import OrderedDict
from ml_ner.feature_extraction.feature_extractor import FeatureExtractor
from ml_ner.corpus.corpusreader import CorpusReader



class ArffAndSciKitDataCreator:
    '''
    Class to create an arff filefrom a given list of features and values created by the corpus_reader and feature_extractor
    '''

    def __init__(self, samples, attributes = []):
        self.samples = samples
        self.attributes = attributes


    def define_features(self):
        '''
        define the list of attributes used to create the arff file
        :return: list of all attributes
        '''

        # List of unbalanced classes
        # classes = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT','EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY','QUANTITY', 'ORDINAL', 'CARDINAL']
        # Balanced list of classes
        classes = ['PERSON', 'NORP', 'ORG', 'GPE', 'DATE', 'PERCENT', 'MONEY', 'CARDINAL']

        attributes_list = []

        for attribute in self.attributes:
            if attribute == 'class':
                attributes_list.append((attribute, classes))
            elif "," in attribute or "%" in attribute:
                attributes_list.append(("'" + attribute + "'", 'REAL'))
            else:
                attributes_list.append(('"' + attribute + '"', 'REAL'))

        return attributes_list



    def generate_arff(self, filename):
        '''
        generate the arff file
        :param filename: defined output filename
        :return:
        '''

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

    def createScikitData(self):
        '''
        generate the feature vectors per sample as a list
        generate the labels per sample a list
        :return: list of feature vectors, list of labels
        '''

        features_list = []
        label_list = []

        for sample in self.samples:
            temp_feature_list = []
            for key,value in sample.items():
                if key != 'class':
                    temp_feature_list.append(value)
                else:
                    label_list.append(value)

            features_list.append(temp_feature_list)

        return features_list, label_list


