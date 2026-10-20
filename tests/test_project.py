import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier import SpamClassifier
from train import read_data, split, evaluate, ROOT


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_data(ROOT / 'data/demo.csv')
        self.training, self.testing = split(self.rows)
        self.model = SpamClassifier().fit(self.training)

    def test_split_has_no_overlap(self):
        self.assertFalse(set(self.training) & set(self.testing))
        self.assertEqual(split(self.rows), split(self.rows))

    def test_vocabulary_excludes_test_only_words(self):
        model = SpamClassifier().fit([('meeting project', 'ham'), ('prize money', 'spam')])
        self.assertNotIn('unseen', model.idf)
        self.assertEqual(model.predict('unseen')['known_terms'], 0)

    def test_direction_and_roundtrip(self):
        model = SpamClassifier().fit([('meeting project schedule', 'ham'), ('win prize money free', 'spam')])
        self.assertEqual(model.predict('win free prize')['label'], 'spam')
        self.assertEqual(model.predict('project meeting schedule')['label'], 'ham')
        restored = SpamClassifier()
        restored.__dict__.update(json.loads(json.dumps(model.__dict__)))
        self.assertEqual(model.predict('win prize'), restored.predict('win prize'))

    def test_empty_and_metrics(self):
        with self.assertRaises(ValueError):
            self.model.predict('  ')
        metrics = evaluate(self.model, self.testing)
        for key in ('accuracy', 'spam_precision', 'spam_recall', 'spam_f1'):
            self.assertTrue(0 <= metrics[key] <= 1)


if __name__ == '__main__':
    unittest.main()
