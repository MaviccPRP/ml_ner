#! /usr/bin/python3
# Phillip Richter-Pechanski
# Created on 02.12.2016
# Example:
#
# Build an instance of the Conll_Extraction Reader Class.
# cr = CorpusReader("/home/tom/Dokumente/UniWiSe2016/ML/ml_ner/wsj/")
# ne = cr.extract_labeled_named_entities()

import os
import re
from itertools import tee, islice, chain
    

class CorpusReader:
    '''
    Corpus Reader for the Ontonotes 2012 conll corpus
    '''
    def __init__(self, path, auto_gold='gold'):
        '''
        Constructor for the CorpusReader class
        :param path: the path to the ontonotes corpus
        :param auto_gold: use auto or gold conll files
        '''
        self.path = path
        self.auto_gold = auto_gold

    def extract_labeled_named_entities(self):
        '''
        Extract all named entities and assign them to their groups.
        Returns a list of dictionaries containing the NEs, its POS tags and its NE group
        Example output:  {'PERSON': [['Mikhail', 'NNP'], ['Gorbachev', 'NNP'], 'NP']}
        '''

        def previous_and_next(some_iterable):
            prevs, items, nexts = tee(some_iterable, 3)
            prevs = chain([None], prevs)
            nexts = chain(islice(nexts, 1, None), [None])
            return zip(prevs, items, nexts)
        
        
        n_entities = []
        save = False
        current_entity = ""
        for root, dirs, files in sorted(os.walk(self.path)):
            path = root.split('/')
            for file in sorted(files):
                if self.auto_gold+'_conll' in file:
                    with open(root + '/' + file, "r") as f:
                        lines = f.readlines()
                        for previous_line, line, next_line in previous_and_next(lines):
                            line = line.split()
                            if previous_line:
                                previous_line = previous_line.split()
                            if previous_line is None or len(previous_line) == 0:
                                previous_line = [""] * 5
                            if next_line:
                                next_line = next_line.split()
                            if next_line is None or len(next_line) == 0:
                                next_line = [""] * 5

                            if len(line) > 0 and '#' not in line[0]:
                                entity = line[10].replace('*','').replace('(','').replace(')','')
                                #if entity == 'GPE' or entity == 'NORP':
                                #    entity = 'GPE_NORP'
                                #if entity == 'PERCENT' or entity == 'CARDINAL' or entity == 'MONEY':
                                #    entity = 'PERCENT_CARDINAL_MONEY'
                                # Get the phrase of the current ne
                                phrase = line[5]  # .replace('*', '').replace('(', '').replace(')', '')
                                reg = re.search("[A-Z][A-Z][A-Z]?[A-Z]?\*", phrase)

                                # If the phrase is starting save it
                                if reg:
                                    reg_phrase = reg.group().replace("*", "")

                                # start extracting ne
                                if '(' in line[10] and save is False:
                                    entity_list = {}
                                    entity_list[entity] = []
                                    entity_list[entity].append([line[3], line[4]])
                                    # Save the helper variables
                                    save = True
                                    current_phrase = reg_phrase
                                    previous_word = previous_line[3]
                                    current_entity = entity
                                # stop if ne is just one word
                                if ')' in line[10] and '*)' not in line[10] and save is True:
                                    entity_list[entity].append(current_phrase)
                                    entity_list[entity].append((previous_word, next_line[3]))
                                    if current_entity not in ["FAC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE","TIME","QUANTITY","ORDINAL","LOC"]:
                                        n_entities.append(entity_list)
                                    save = False
                                # add multiple words to the current ne
                                if line[10] is '*' and save is True:
                                    entity_list[current_entity].append([line[3], line[4]])
                                    continue
                                # close the multiline ne
                                if '*)' in line[10] and save is True:
                                    entity_list[current_entity].append([line[3], line[4]])
                                    entity_list[current_entity].append(current_phrase)
                                    entity_list[current_entity].append((previous_word, next_line[3]))
                                    if current_entity not in ["FAC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE","TIME","QUANTITY","ORDINAL","LOC"]:
                                        n_entities.append(entity_list)
                                    save = False

        return n_entities
