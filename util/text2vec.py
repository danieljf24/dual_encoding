import numpy as np

from basic.bigfile import BigFile
from basic.constant import logger
from util.vocab import clean_str

INFO = __file__


class Text2Vec(object):
    """ Text2Vec """
    def __init__(self, datafile, ndims=0, L1_norm=0, L2_norm=0):
        logger.info(INFO + '.' + self.__class__.__name__+ ' initializing ...')
        self.datafile = datafile
        self.nidms = ndims
        self.L1_norm = L1_norm
        self.L2_norm = L2_norm

        assert (L1_norm + L2_norm) <= 1

    def preprocess(self, query, clear):
        if clear:
            words = clean_str(query)
        else:
            words = query.strip().split()
        return words

    def do_L1_norm(self, vec):
        L1_norm = np.linalg.norm(vec, 1)
        return 1.0 * np.array(vec) / L1_norm

    def do_L2_norm(self, vec):
        L2_norm = np.linalg.norm(vec, 2)
        return 1.0 * np.array(vec) / L2_norm

    def embedding(self, query):
        vec = self.mapping(query)
        if vec is not None:
            vec = np.array(vec)
        return vec

# Vocab
class Bow2Vec(Text2Vec):

    def __init__(self, vocab, ndims=0, L1_norm=0, L2_norm=0):
        super(Bow2Vec, self).__init__(vocab, ndims, L1_norm, L2_norm)

        self.vocab = vocab
        if ndims != 0:
            assert(len(self.vocab) == ndims) , \
                "feature dimension not match %d != %d" % (len(self.vocab), self.ndims)
        else:
            self.ndims = len(self.vocab)

    def mapping(self, query, clear=True):
        words = self.preprocess(query, clear)

        vec = [0.0]*self.ndims

        for word in words:
            if word in self.vocab.word2idx:
                vec[self.vocab(word)] += 1

        if sum(vec) > 0:

            if self.L1_norm:
                return self.do_L1_norm(vec)
            if self.L2_norm:
                return self.do_L2_norm(vec)

            return np.array(vec)

        else:
            return None



class AveWord2Vec(Text2Vec):

    def __init__(self, datafile, ndims=0, L1_norm=0, L2_norm=0):
        super(AveWord2Vec, self).__init__(datafile, ndims, L1_norm, L2_norm)

        self.word2vec = BigFile(datafile)
        if ndims != 0:
            assert(self.word2vec.ndims == ndims) , \
                "feature dimension not match %d != %d" % (self.word2vec.ndims, self.ndims)
        else:
            self.ndims = self.word2vec.ndims

    def mapping(self, query, clear=True):
        words = self.preprocess(query, clear)

        renamed, vectors = self.word2vec.read(words)

        if len(renamed) != len(words):
            renamed2vec = dict(zip(renamed, vectors))
            vectors = []
            for word in words:
                if word in renamed2vec:
                    vectors.append(renamed2vec[word])


        if len(vectors) > 0:
            vec = np.array(vectors).mean(axis=0)

            if self.L1_norm:
                return self.do_L1_norm(vec)
            if self.L2_norm:
                return self.do_L2_norm(vec)

            return vec
        else:
            return None



NAME_TO_ENCODER = {'word2vec': AveWord2Vec, 'bow': Bow2Vec}


def get_text_encoder(name):
    assert name in NAME_TO_ENCODER
    return NAME_TO_ENCODER[name]


if __name__ == '__main__':
    t2v = AveWord2Vec('/home/daniel/VisualSearch/word2vec/flickr/vec500flickr30m')
