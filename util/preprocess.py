from collections import Counter, defaultdict
from itertools import islice

def preprocess(documents):
    unigrams = {}
    bigrams = {}
    for doc_id, text in documents.items():
        words = text.lower().split()
        unigrams[doc_id] = Counter(words)
        bigrams[doc_id] = Counter(zip(words, islice(words, 1, None)))
    return unigrams, bigrams
