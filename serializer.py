from sklearn.externals import joblib


def save(categorizer, filename):
    joblib.dump(categorizer, filename)


def load(filename):
    return joblib.load(filename)
