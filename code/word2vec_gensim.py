# -*- coding: utf-8 -*-

import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    print sys.argv[1:3]
    inp, outp1=sys.argv[1:3]
    size =int(sys.argv[3])
    iter=int(sys.argv[4])


    model = Word2Vec(LineSentence(inp), sg=1, size=size, window=5, min_count=5,iter=iter,
                     workers=multiprocessing.cpu_count())
    model.save(outp1)



