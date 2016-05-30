import string
import numpy as np

from collections import Counter, namedtuple
from base import BaseRPBuilder
from wikitools.wiki import Wiki
from wikitools.page import Page
from wikitools.category import Category
from wikitools.api import APIRequest
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WordPunctTokenizer


BASE_URL = 'https://en.wikipedia.org/w/api.php'
PAGE_NS = 0
CATEGORY_NS = 14

SearchResult = namedtuple('SearchResult', 'title summary url')


class WikiRPBuilder(BaseRPBuilder):

    def __init__(self, categories):
        BaseRPBuilder.__init__(self, categories)
        self.wiki = Wiki(BASE_URL)

    def build_rp(self, category):
        wiki_cats = self.search_category(category)
        if wiki_cats:
            wiki_cat = wiki_cats[0]
        else:
            p = self.search_page(category)[0]
            page = Page(self.wiki, p.title)
            wiki_cat = Category(self.wiki, page.getCategories()[0])
        # Only take first category for now, relies on wiki search being good
        related_concepts = self.get_category_members(wiki_cat.title, depth=2)
        counter = self.analyse_category(related_concepts)
        words = map(list, zip(*counter.most_common(100)))[0]
        return words

    def search_page(self, page):
        params = {
            'action': 'opensearch',
            'search': page,
            'namespace': PAGE_NS,
            'suggest': True,
            'redirects': 'resolve',
            'format': 'jsonfm',
        }
        r = APIRequest(self.wiki, params)
        results = self._parse_search_results(r.query())
        return results

    def search_category(self, category):
        params = {
            'action': 'opensearch',
            'search': category,
            'namespace': CATEGORY_NS,
            'suggest': True,
            'redirects': 'resolve',
            'format': 'jsonfm',
        }
        r = APIRequest(self.wiki, params)
        results = self._parse_search_results(r.query())
        return results

    def _parse_search_results(self, raw_result):
        results = np.array(raw_result[1:])
        if len(results) == 0:
            return results
        return [SearchResult(*results[:, i]) for i in xrange(len(results[0]))]

    def get_category_members(self, category_name, depth=1):
        print category_name, depth
        members = []
        c = Category(self.wiki, category_name)
        for m in c.getAllMembersGen():
            if m.title[:9] == 'Category:':
                # print m.title, depth
                if depth > 0:
                    members.extend(self.get_category_members(m.title, depth - 1))
            else:
                members.append(m)
        return members

    def analyse_category(self, members):
        tknzr = WordPunctTokenizer()
        stemmer = PorterStemmer()
        count = Counter()
        for title in [m.title for m in members]:
            try:
                title = str(title).lower().translate(None, string.punctuation)
                for word in tknzr.tokenize(title):
                    w = stemmer.stem(word)
                    if w not in self.stops:
                        count[w] += 1
            except UnicodeEncodeError:
                pass
        return count
