import math


def compute_tf(term, term_counts):
    return term_counts[term] / sum(term_counts.values())

def compute_idf(term, all_docs):
    doc_count = sum(1 for doc in all_docs if term in doc)
    return math.log((1 + len(all_docs)) / (1 + doc_count)) + 1

def tf_idf(query, unigrams, idf_values):
    scores = {}
    for doc_id, term_counts in unigrams.items():
        score = sum(compute_tf(term, term_counts) * idf_values.get(term, 0) for term in query)
        scores[doc_id] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

def bm25(query, unigrams, idf_values, k1=1.5, b=0.75):
    scores = {}
    avg_doc_len = sum(sum(counts.values()) for counts in unigrams.values()) / len(unigrams)
    for doc_id, term_counts in unigrams.items():
        doc_len = sum(term_counts.values())
        score = 0
        for term in query:
            tf = term_counts[term]
            idf = idf_values.get(term, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf * numerator / denominator
        scores[doc_id] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]



def main():
    print("hello")

if __name__ == "__main__":
    main()
