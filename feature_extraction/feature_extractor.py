#! /usr/bin/python3
# Phillip Richter-Pechanski
# Created on 02.12.2016
# Example:

class FeatureExtractor:

    def __init__(self, ne_list):
        self.ne_list = ne_list

    def extract_features_baseline(self):
        return self.ne_list