from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer

from labeller import DocumentLabeller
from rp.wiki import WikiRPBuilder
# from rp import build_rps


def tokenize(tweet):
    tknzr = TweetTokenizer(strip_handles=True)
    stemmer = PorterStemmer()
    tokens = tknzr.tokenize(tweet)
    return [stemmer.stem(t) for t in tokens]


class AutoCategorizer:

    def __init__(self, categories, docs, classifier, cached_rps=None,
                 cached_vocab=None, cached_d_v=None, cached_r_v=None):
        self.categories = categories
        self.docs = docs
        if cached_rps is not None:
            print '> Using cached rps'
            self.rps = cached_rps
        else:
            print '> Building rps'
            rp_builder = WikiRPBuilder(categories)
            self.rps = rp_builder.build_rps()

        if cached_vocab is not None:
            print '> Using cached vocab'
            self.vocab = cached_vocab
        else:
            print '> Collating vocab'
            self.vocab = self.collate_vocab([docs, self.rps])

        if cached_d_v is not None:
            print '> Using cached docs vectorizer'
            self.docs_vectorizer = cached_d_v
            self.docs_features = self.docs_vectorizer.transform(self.docs)
        else:
            print '> Building docs feature matrices'
            self.docs_vectorizer = self.build_vectorizer(self.vocab)
            self.docs_features = self.docs_vectorizer.fit_transform(self.docs)

        if cached_r_v is not None:
            print '> Using cached rps vectorizer'
            self.rps_vectorizer = cached_r_v
            self.rps_features = self.rps_vectorizer.transform(self.rps)
        else:
            print '> Building rps feature matrices'
            self.rps_vectorizer = self.build_vectorizer(self.vocab)
            self.rps_features = self.rps_vectorizer.fit_transform(self.rps)

        self.classifier = classifier
        self.categorizer = self.build_categorizer()

    def build_categorizer(self):
        print '> Labelling documents using FACT'
        dl = DocumentLabeller(self.docs_features, self.rps_features)
        training_data, targets = dl.label()
        print '> {}, {}'.format(training_data.shape, targets.shape)

        self.dl = dl

        print '> Training classifier using labelled docs'
        categorizer = OneVsRestClassifier(self.classifier)
        categorizer.fit(training_data, targets)
        return categorizer

    def collate_vocab(self, all_docs):
        vocab = set()

        for docs in all_docs:
            for doc in docs:
                for token in tokenize(doc):
                    vocab.add(token)

        return vocab

    def build_vectorizer(self, vocab):
        tfidf = TfidfVectorizer(
            tokenizer=tokenize,
            stop_words='english',
            decode_error='ignore',
            vocabulary=vocab
        )
        return tfidf

    def classify(self, doc):
        feature_vec = self.docs_vectorizer.transform([doc])
        return self.categorizer.predict(feature_vec)
