"""Validate CSV, split before fitting, evaluate, and save a JSON model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
from classifier import SpamClassifier, tokens

ROOT = Path(__file__).resolve().parent


def read_data(path):
    unique = {}
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'text', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV must have text,label columns.')
        for number, row in enumerate(reader, 2):
            text, label = (row.get('text') or '').strip(), (row.get('label') or '').strip().lower()
            key = ' '.join(tokens(text))
            if not key or label not in ('ham', 'spam'):
                raise ValueError(f'Invalid text or label on CSV row {number}.')
            if key in unique and unique[key][1] != label:
                raise ValueError(f'Conflicting labels on row {number}.')
            unique[key] = (text, label)
    rows = list(unique.values())
    if any(sum(label == c for _, label in rows) < 5 for c in ('ham', 'spam')):
        raise ValueError('Provide at least five distinct emails per class.')
    return rows


def split(rows, seed=42):
    rng = random.Random(seed)
    train, test = [], []
    for label in ('ham', 'spam'):
        group = sorted([r for r in rows if r[1] == label])
        rng.shuffle(group)
        count = max(1, round(len(group) * .2))
        test.extend(group[:count])
        train.extend(group[count:])
    return train, test


def evaluate(model, rows):
    matrix = {actual: {pred: 0 for pred in ('ham', 'spam')} for actual in ('ham', 'spam')}
    for text, actual in rows:
        matrix[actual][model.predict(text)['label']] += 1
    tp, fp = matrix['spam']['spam'], matrix['ham']['spam']
    fn, tn = matrix['spam']['ham'], matrix['ham']['ham']
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {'accuracy': (tp + tn) / len(rows), 'spam_precision': precision, 'spam_recall': recall,
            'spam_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0,
            'confusion_matrix_actual_then_predicted': matrix}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/demo.csv')
    args = parser.parse_args()
    rows = read_data(args.data)
    training, testing = split(rows)
    model = SpamClassifier().fit(training)
    metrics = evaluate(model, testing)
    metrics.update({'training_rows': len(training), 'test_rows': len(testing), 'seed': 42,
                    'dataset': args.data.name, 'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    'evaluation': '80/20 stratified holdout; vocabulary fitted only on training rows',
                    'limitation': 'Small synthetic demo results do not measure real-world performance.' if args.data.resolve() == (ROOT / 'data/demo.csv').resolve() else 'Single random holdout; audit source, near-duplicates, temporal shift and privacy before use.'})
    out = ROOT / 'artifacts'
    out.mkdir(exist_ok=True)
    # Keep the evaluated training-only model; do not silently retrain on test data.
    (out / 'model.json').write_text(json.dumps({'idf': model.idf, 'priors': model.priors, 'weights': model.weights, 'metadata': metrics}, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
