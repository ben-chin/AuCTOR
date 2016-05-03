from collections import namedtuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

stops = set(stopwords.words('english'))
RP = namedtuple('RP', 'concept hypernyms hyponyms meronyms holonyms drfs')


def build_rp(category):
    features = get_seed_features(category)
    concepts = get_concepts(features)

    rps = []
    for concept in concepts:
        expansion = get_partial_profile(concept)
        rps.extend(flatten_expansion(expansion))

    return rps


def get_seed_features(category):
    tokens = word_tokenize(category)
    features = [w for w in tokens if w not in stops]
    return features


# Filtered concepts with compromised policy and
# gloss vector similarity measure
# TODO: actually filter
def get_concepts(features):
    concepts = []
    for f in features:
        concepts.extend(wn.synsets(f))
    return concepts


def get_partial_profile(concept):
    hypernyms = get_hypernyms(concept)
    hyponyms = get_hyponyms(concept)
    meronyms = get_meronyms(concept)
    holonyms = get_holonyms(concept)
    drfs = get_derivationally_related_forms(concept)
    return RP(concept, hypernyms, hyponyms, meronyms, holonyms, drfs)


def clean_lemma_names(names):
    ns = [' '.join(n.split('_')) for n in names]
    return [n for n in ns if n not in stops]


def flatten_expansion(profile):
    return get_synsets_words([profile.concept])
    + get_synsets_words(profile.hypernyms)
    + get_synsets_words(profile.hyponyms)
    + get_synsets_words(profile.meronyms)
    + get_synsets_words(profile.holonyms)
    + get_synsets_words(profile.drfs)


def get_synsets_words(synsets):
    words = []
    for s in synsets:
        words.extend(clean_lemma_names(s.lemma_names()))
    return words


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
    return drfs
