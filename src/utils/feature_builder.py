#!/usr/bin/env python

import numpy
from nltk.util import ngrams
from str_manip import StrManip as StrManip

class FeatureBuilder(StrManip):
    def __init__(self):
        super().__init__()
        self.vocab = None

    def compile_vocabulary(self, data, use_bigrams=False):
        '''
        Compiles vocabulary from given data.

        @param data: single column dataframe containing text
        @param use_bigrams: include bigrams in vocabulary if True
        @return: None
        '''
        v = []

        for row in data:
            words = self.regex_tokenizer(row)

            v += [w for w in words]

            if use_bigrams:
                v += [' '.join(b) for b in ngrams(words, 2)]

        v = set(v)
        v = list(v)
        v.sort()

        self.vocab = v
    
    def get_features(self, data, use_bigrams=False):
        '''
        Creates features from the given data.

        @param data: raw data to be processed
        @param use_bigrams: include bigrams in the features
        @return: document-by-feature matrix
        '''
        if self.vocab is None: self.compile_vocabulary(data, use_bigrams)
        
        feats = numpy.zeros((len(self.vocab), len(data)))

        for i, d in enumerate(data):
            words = self.regex_tokenizer(d)

            for w in words:
                if w in self.vocab: feats[self.vocab.index(w)][i] = 1

            if use_bigrams:
                bigrams = ngrams(words, 2)
                
                for b in bigrams:
                    if b in self.vocab: feats[self.vocab.index(b)][i] = 1

        return feats.T
