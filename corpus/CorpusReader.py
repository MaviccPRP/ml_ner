#! /usr/bin/python3
# Phillip Richter-Pechanski
# Created on 02.12.2016
# Example:
#
# Build an instance of the Conll_Extraction Reader Class.
# cr = CorpusReader("/home/tom/Dokumente/UniWiSe2016/ML/ml_ner/wsj/")
# ne = cr.extract_labeled_named_entities()

import os

class CorpusReader:
    '''
    Corpus Reader for the Ontonotes 2012 conll corpus
    '''
    def __init__(self, path):
        '''
        Constructor for the CorpusReader class
        :param path: the path to the ontonotes corpus
        '''
        self.path = path

    def extract_labeled_named_entities(self):
        '''
        Extract all named entities and assign them to their groups.
        Returns a list of dictionaries containing the NEs, its POS tags and its NE group
        Example output:  {'PERSON': [['Mikhail', 'NNP'], ['Gorbachev', 'NNP']]}
        '''
        n_entities = []
        save = False
        current_entity = ""
        for root, dirs, files in os.walk(self.path):
            path = root.split('/')
            for file in files:
                if 'gold_conll' in file:
                    with open(root + '/' + file, "r") as f:
                        for line in f:
                            line = line.split()
                            if len(line) > 0 and '#' not in line[0]:
                                entity = line[10].replace('*','').replace('(','').replace(')','')
                                # start extracting ne
                                if '(' in line[10] and save is False:
                                    entity_list = {}
                                    entity_list[entity] = []
                                    entity_list[entity].append([line[3], line[4]])
                                    save = True
                                    current_entity = entity
                                # stop if ne is just one word
                                if ')' in line[10] and '*)' not in line[10] and save is True:
                                    n_entities.append(entity_list)
                                    save = False
                                # add multiple words to the current ne
                                if line[10] is '*' and save is True:
                                    entity_list[current_entity].append([line[3], line[4]])
                                    continue
                                # close the multiline ne
                                if '*)' in line[10] and save is True:
                                    entity_list[current_entity].append([line[3], line[4]])
                                    n_entities.append(entity_list)
                                    save = False

        return n_entities
