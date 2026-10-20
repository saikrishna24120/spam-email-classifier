"""TF-IDF + multinomial Naive Bayes, using only the standard library."""
import collections
import math
import re


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class SpamClassifier:
    def fit(self, rows):
        if {label for text, label in rows} != {'ham', 'spam'}:
            raise ValueError('Training data must contain ham and spam.')
        df = collections.Counter()
        for text, _ in rows:
            df.update(set(tokens(text)))
        if not df:
            raise ValueError('Training text must contain words.')
        self.idf = {t: math.log((1 + len(rows)) / (1 + n)) + 1 for t, n in df.items()}
        counts = {c: collections.Counter() for c in ('ham', 'spam')}
        docs = collections.Counter(label for _, label in rows)
        for text, label in rows:
            counts[label].update(self.vector(text))
        self.priors = {c: math.log(docs[c] / len(rows)) for c in counts}
        self.weights = {}
        for c, values in counts.items():
            denominator = sum(values.values()) + len(self.idf)
            self.weights[c] = {t: math.log((values[t] + 1) / denominator) for t in self.idf}
        return self

    def vector(self, text):
        values = {t: n * self.idf[t] for t, n in collections.Counter(tokens(text)).items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in values.values())) or 1
        return {t: v / norm for t, v in values.items()}

    def predict(self, text):
        if not text.strip():
            raise ValueError('Enter an email first.')
        vector = self.vector(text)
        scores = {c: self.priors[c] + sum(v * self.weights[c][t] for t, v in vector.items()) for c in self.weights}
        delta = max(-700, min(700, scores['ham'] - scores['spam']))
        probability = 1 / (1 + math.exp(delta))
        evidence = sorted(((t, v * (self.weights['spam'][t] - self.weights['ham'][t])) for t, v in vector.items()), key=lambda pair: abs(pair[1]), reverse=True)[:8]
        return {'label': 'spam' if probability >= 0.5 else 'ham', 'spam_score': probability,
                'known_terms': len(vector), 'evidence': evidence}
