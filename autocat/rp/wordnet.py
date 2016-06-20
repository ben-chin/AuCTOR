from collections import namedtuple
from operator import itemgetter
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from base import BaseRPBuilder
from sense_ranker import SenseRanker


PartialRP = namedtuple(
    'PartialRP',
    'concept hypernyms hyponyms meronyms holonyms drfs'
)


def get_hypernyms(concept):
    return concept.hypernyms()


def get_hyponyms(concept):
    return concept.hyponyms()


def get_meronyms(concept):
    return concept.part_meronyms()
    + concept.substance_meronyms()
    + concept.member_meronyms()


def get_holonyms(concept):
    return concept.part_holonyms()
    + concept.substance_holonyms()
    + concept.member_holonyms()


def get_derivationally_related_forms(concept):
    drfs = []
    for l in concept.lemmas():
        drfs.extend(l.derivationally_related_forms())
    return [d.synset() for d in drfs]


class WordNetRPBuilder(BaseRPBuilder):

    def __init__(self, categories):
        BaseRPBuilder.__init__(self, categories)
        self.sense_ranker = SenseRanker(categories)

    def build_rp(self, category):
        features = self.get_seed_features(category)
        concepts = self.get_concepts(features)

        rp = []
        for concept in concepts:
            expansion = self.get_partial_profile(concept)
            rp.extend(self.flatten_expansion(expansion))

        return rp

    def get_seed_features(self, category):
        tokens = word_tokenize(category)
        features = [w for w in tokens if w not in self.stops]
        return features

    def get_concepts(self, features):
        concepts = []
        for f in features:
            potential_senses = wn.synsets(f)
            senses = self.extract_best_senses(f, potential_senses)
            concepts.extend(senses)
        return concepts

    def extract_best_senses(self, word, senses):
        get_sense, get_rank = itemgetter(0), itemgetter(1)
        s_ranks = [(s, self.sense_ranker.get_rank(word, s)) for s in senses]
        s_ranks.sort(reverse=True, key=get_rank)
        threshold = max(1, int(0.4 * len(senses)))  # parameterise
        return map(get_sense, s_ranks[: threshold])

    def get_partial_profile(self, concept):
        hypernyms = get_hypernyms(concept)
        hyponyms = get_hyponyms(concept)
        meronyms = get_meronyms(concept)
        holonyms = get_holonyms(concept)
        drfs = get_derivationally_related_forms(concept)
        return PartialRP(
            concept,
            hypernyms,
            hyponyms,
            meronyms,
            holonyms,
            drfs
        )

    def clean_lemma_names(self, names):
        ns = [' '.join(n.split('_')) for n in names]
        return [n for n in ns if n not in self.stops]

    def get_synsets_words(self, synsets):
        words = []
        for s in synsets:
            words.extend(self.clean_lemma_names(s.lemma_names()))
        return words

    def flatten_expansion(self, profile):
        return self.get_synsets_words([profile.concept]) \
            + self.get_synsets_words(profile.hypernyms) \
            + self.get_synsets_words(profile.hyponyms) \
            + self.get_synsets_words(profile.meronyms) \
            + self.get_synsets_words(profile.holonyms) \
            + self.get_synsets_words(profile.drfs)
