from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer

from labeller import DocumentLabeller
from rp import build_rp


def tokenize(tweet):
    tknzr = TweetTokenizer(strip_handles=True)
    stemmer = PorterStemmer()
    tokens = tknzr.tokenize(tweet)
    return [stemmer.stem(t) for t in tokens]


class AutoCategorizer:

    def __init__(self, categories, docs, classifier):
        print '> Building rps'
        self.categories = categories
        self.rps = self.build_rps(categories)
        self.docs = docs

        print '> Collating vocab'
        vocab = self.collate_vocab([docs, self.rps])
        self.docs_vectorizer = self.build_vectorizer(vocab)
        self.rps_vectorizer = self.build_vectorizer(vocab)

        print '> Building feature matrices'
        self.docs_features = self.docs_vectorizer.fit_transform(self.docs)
        self.rps_features = self.rps_vectorizer.fit_transform(self.rps)

        self.classifier = classifier
        self.categorizer = self.build_categorizer()

    def build_categorizer(self):
        print '> Labelling documents using FACT'
        dl = DocumentLabeller(self.docs_features, self.rps_features)
        training_data, targets = dl.label()
        print training_data.shape, targets.shape

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

    def build_rps(self, categories):
        return [' '.join(build_rp(c)) for c in categories]

    def classify(self, doc):
        feature_vec = self.docs_vectorizer.transform([doc])
        return self.categorizer.predict(feature_vec)
