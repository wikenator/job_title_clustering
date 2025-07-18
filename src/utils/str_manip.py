#!/usr/bin/env python

import sys
import pickle
from nltk import RegexpTokenizer as RT
from nltk.corpus import stopwords

class StrManip():
    def __init__(self):
        # create array of selected stop words and punctuation
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set([',', '.', '?', ':', ';', '$', '-', '|', '+', '&', '(', ')', '/'])

        for punc in self.punctuation:
            self.stop_words.add(punc)
            
        self.stop_words.remove('it') # short for information technology
    
    def regex_tokenizer(self, text):
        '''
        Lowercase text, tokenize text, and remove stopwords.

        @param text: raw text to tokenize
        @return: tokenized text minus stopwords
        '''

        text = text.lower()
        regex_tokenizer = RT('\w+|[\\' + '\\'.join(self.punctuation) + ']|\S+')
        tokens = regex_tokenizer.tokenize(text)

        return [t for t in tokens if not t in self.stop_words]
    