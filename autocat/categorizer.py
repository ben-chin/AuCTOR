from sklearn.multiclass import OneVsRestClassifier

from labeller import DocumentLabeller


class AutoCategorizer:

    def __init__(self, categories, docs,
                 feature_pipeline,
                 rp_builder, cached_rps=None):
        self.docs = docs
        self.categories = categories
        self.feature_pipeline = feature_pipeline
        self.training_docs = None

        # Get Representative Profiles
        if cached_rps is not None:
            print '> Using cached rps'
            self.rps = cached_rps
        else:
            print '> Building rps'
            self.rps = rp_builder(categories).build_rps()

        self.categorizer = self.build_categorizer(feature_pipeline)

    def build_categorizer(self, feature_pipeline):
        print '> Init DocumentLabeller'
        dl = DocumentLabeller(self.docs, self.rps)

        print '> Labelling documents using FACT'
        docs, targets = dl.label()
        self.training_docs = docs
        print '> {}, {}'.format(len(self.training_docs), targets.shape)

        print '> Training classifier using labelled docs'
        categorizer = OneVsRestClassifier(feature_pipeline)
        categorizer.fit(self.training_docs, targets)
        return categorizer

    def get_training_docs(self):
        return self.training_docs

    def classify(self, doc):
        return self.categorizer.predict([doc])

    def classify_many(self, docs):
        return self.categorizer.predict(docs)
