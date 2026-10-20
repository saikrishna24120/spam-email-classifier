# Mail Lens — Spam Email Classifier

A complete educational Python project: browser interface, TF-IDF feature extraction, multinomial Naive Bayes, CSV training, saved JSON model, reproducible evaluation, tests, and a project report. Python 3.10+; no third-party packages required.

## Quick start (Windows, macOS, Linux)

Open a terminal in this folder:

```sh
python train.py
python app.py
```

Visit http://127.0.0.1:8000 in a browser. On Windows, use `py` instead of `python` if your installation provides the Python launcher. Press Ctrl+C to stop the server. If port 8000 is occupied, stop the other service or change the port in app.py.

Paste an email subject and body, then select **Classify email**. The app shows a label, a spam model score, and the terms that most influenced the score. Requests stay on your local machine and email text is not saved. There is no mailbox connection or automatic deletion.

## Train on your own email dataset

Create a UTF-8 CSV with `text,label` columns. Quote fields containing commas or line breaks. Labels must be `ham` or `spam`; provide at least five distinct examples in each class. Use hundreds or thousands of diverse, appropriately sourced emails for a meaningful experiment.

```csv
text,label
"Subject: Meeting. Please review tomorrow's agenda.",ham
"Subject: Winner! Claim your free cash prize now.",spam
```

```sh
python train.py --data data/my_emails.csv
python -m unittest discover -s tests -v
python app.py
```

Restart the app after retraining. Training overwrites artifacts/model.json and artifacts/metrics.json. The deployed model is the same training-only model used for evaluation. JSON avoids executable pickle loading; only load model files you trust.

## Included data and evaluation

`data/demo.csv` contains 50 original, manually authored synthetic messages, evenly split between ham and spam. They are deliberately simple demonstration data, not a research email corpus. No external dataset is bundled. The report and evaluation explicitly preserve this limitation.

Training normalizes exact duplicate text for deduplication, rejects conflicting labels, and uses a seeded stratified 80/20 holdout. The vocabulary and IDF are fitted on training emails only. Evaluation reports accuracy, spam precision, recall, F1, and a confusion matrix. The JSON includes dataset hash, row counts, and seed.

Random splitting does not prevent leakage from near-duplicate campaigns, shared senders, or related threads. For a real study use a separate time-based test set and campaign/thread grouping. Do not tune parameters against the final test set.

## Files

| File | Purpose |
|---|---|
| classifier.py | Tokenization, TF-IDF, Naive Bayes, evidence |
| train.py | CSV validation, training, evaluation, JSON export |
| app.py / index.html | Local HTTP API and responsive browser interface |
| data/demo.csv | Synthetic demonstration dataset |
| artifacts/ | Evaluated model and measured metrics |
| tests/test_project.py | Split, vocabulary isolation, prediction, serialization tests |
| PROJECT_REPORT.md | Methodology, results, limitations, viva questions |

## Limitations

English alphanumeric tokens only. This prototype does not parse MIME, attachments, HTML, sender reputation, or authentication headers. Unknown words are ignored. Scores are uncalibrated; an unfamiliar or adversarial email can be misclassified. Do not use a ham label as evidence that a link or attachment is safe. The built-in server is for local demonstrations and does not include production authentication or hosting controls.

## References

- scikit-learn's [text feature extraction guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) describes TF-IDF.
- scikit-learn's [MultinomialNB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) explains the algorithm family and use of fractional TF-IDF features. This project implements the approach directly in Python.
- Python [http.server documentation](https://docs.python.org/3/library/http.server.html) describes the local server API.
