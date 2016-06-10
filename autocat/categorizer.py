from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from tokenize import tokenize
from labeller import DocumentLabeller
from rp.wiki import WikiRPBuilder


class AutoCategorizer:

    def __init__(self, categories, docs, classifier, cached_rps=None,
                 reduce_features=False):
        self.docs = docs
        self.categories = categories
        self.reduce_features = reduce_features
        self.training_docs = None

        # Get Representative Profiles
        if cached_rps is not None:
            print '> Using cached rps'
            self.rps = cached_rps
        else:
            print '> Building rps'
            rp_builder = WikiRPBuilder(categories)
            self.rps = rp_builder.build_rps()

        self.categorizer = self.build_categorizer(classifier)

    def build_categorizer(self, classifier):
        print '> Init DocumentLabeller'
        dl = DocumentLabeller(self.docs, self.rps)
        print '> Labelling documents using FACT'
        docs, targets = dl.label()
        self.training_docs = docs
        print '> {}, {}'.format(len(self.training_docs), targets.shape)

        print '> Training classifier using labelled docs'
        features = FeatureUnion([
            (
                'bag_of_words', Pipeline([
                    ('count', self.build_vectorizer()),
                    ('tfidf', TfidfTransformer()),
                    # ('dim_reduce', TruncatedSVD(
                    #     n_components=60,
                    #     random_state=42
                    # )),
                ])
            ),
        ])

        categorizer = OneVsRestClassifier(
            Pipeline([
                ('features', features),
                ('selection', SelectFromModel(classifier, threshold=0.2)),
                ('classifier', LinearSVC()),
            ])
        )

        categorizer.fit(self.training_docs, targets)
        return categorizer

    def build_vectorizer(self):
        return HashingVectorizer(
            tokenizer=tokenize,
            ngram_range=(1, 2),  # doesn't make sense for rps
            stop_words='english',
            decode_error='ignore',
        )

    def get_training_docs(self):
        return self.training_docs

    def classify(self, doc):
        return self.categorizer.predict([doc])
