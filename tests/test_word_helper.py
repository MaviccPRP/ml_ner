# This script tests the word helper method to extract words in a corpus
# Useage: python test_wordhelper.py > test_wordhelper_output.txt
# The results will be printed to test_wordhelper_output.txt

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pprint
from ml_ner.feature_extraction.word_helpers import word_helper

print(word_helper())


