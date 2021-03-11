# -*- coding: utf-8 -*-

import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word_vectors = KeyedVectors.load_word2vec_format('../data/glove.6B/glove.6B.200d.txt', binary=False)
    #
    # model = Word2Vec()
    #
    # model.save(outp1)
