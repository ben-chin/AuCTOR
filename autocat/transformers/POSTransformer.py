from collections import defaultdict
from nltk import word_tokenize, pos_tag
from sklearn.base import BaseEstimator, TransformerMixin


class POSTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, tokenize=None, binary=False, verbose=False):
        self.binary = binary
        self.tokenize = tokenize if tokenize else word_tokenize
        self.features = defaultdict(int) if binary else {}
        self.verbose = verbose

    def fit(self, X, y=None):
        print '>>>> Fitting docs'
        for doc in X:
            tokens = self.tokenize(doc)
            tags = pos_tag(tokens)
            if self.binary:
                for tag in tags:
                    self.features[self._tag_to_string(tag)] = True
            else:
                for tag in tags:
                    self.features[self._tag_to_string(tag)] += 1

        print '>>>> Created {} features'.format(len(self.features.keys()))
        return self

    def transform(self, raw_X, y=None):
        print '>>>> Transforming docs'
        results = []
        for doc in raw_X:
            tokens = self.tokenize(doc)
            tags = pos_tag(tokens)
            features = defaultdict(int) if self.binary else {}
            if self.binary:
                for tag in tags:
                    features[self._tag_to_string(tag)] = True
                else:
                    for tag in tags:
                        self.features[self._tag_to_string(tag)] += 1
            results.append(features)

        print '>>>> Finished transforming docs'
        return results

    def fit_transform(self, raw_X, y=None):
        self.fit(raw_X, y)
        return self.transform(raw_X, y)

    def _tag_to_string(self, tag):
        tkn, pos = tag
        return '{}-{}'.format(tkn, pos)
