from nltk.corpus import stopwords


class BaseRPBuilder:

    def __init__(self, categories):
        self.categories = categories
        self.stops = set(stopwords.words('english'))

    def build_rps(self):
        return [' '.join(self.build_rp(c)) for c in self.categories]

    def build_rp(self, category):
        pass
