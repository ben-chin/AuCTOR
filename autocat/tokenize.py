import regex as re

from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer


def tokenize(tweet, stem=True):
    tknzr = TweetTokenizer(
        strip_handles=True,
        preserve_case=True,
        reduce_len=True
    )
    tweet = clean_tweet(tweet)
    tokens = tknzr.tokenize(tweet)

    if stem:
        stemmer = PorterStemmer()
        return [stemmer.stem(t) for t in tokens]
    else:
        return tokens


def clean_tweet(tweet):
    # Strip non-standard chars
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    # Remove urls
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'\#', '', tweet)
    # Remove RT token
    tweet = re.sub(r'RT', '', tweet)
    # Remove punctuation
    tweet = re.sub(ur'\p{P}+', '', tweet)
    # Lowercase
    # tweet = tweet.lower()
    return tweet
