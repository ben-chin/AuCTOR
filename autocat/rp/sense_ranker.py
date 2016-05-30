from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SenseRanker:

    def __init__(self, categories):
        self.categories = categories
        self.word_set = self._get_word_set()
        self.context = self._build_context()
        self.vectorizer = self._build_vectorizer()

    def get_rank(self, word, sense):
        rank = 0.0
        cws = self._get_contextual_words(word)
        for cw in cws:
            for cw_sense in self.context[cw]:
                rank += self.get_r(sense, cw_sense)
        return rank

    def get_r(self, sense_1, sense_2):
        return cosine_similarity(
            self.vectorizer.transform([sense_1.definition()]),
            self.vectorizer.transform([sense_2.definition()])
        )

    def _build_vectorizer(self):
        v = CountVectorizer()
        corpus = self._get_contextual_glosses()
        v.fit_transform(corpus)
        return v

    def _build_context(self):
        ctx = {}
        for word in self.word_set:
            ctx[word] = wn.synsets(word)
        return ctx

    def _get_contextual_glosses(self):
        glosses = []
        for senses in self.context.values():
            glosses.extend([s.definition() for s in senses])
        return glosses

    def _get_contextual_words(self, word):
        return self.word_set.difference(set([word]))

    def _get_word_set(self):
        words = set()
        for c in self.categories:
            # should tokenise here and remove stopwords
            words.update(c.split())
        return words
