# Spam Email Classification Using TF-IDF and Naive Bayes

**Student:** ____________________  
**Institution / department:** ____________________  
**Supervisor:** ____________________  
**Submission date:** ____________________

## Abstract

This project implements an educational spam email classifier using natural language processing and supervised learning. Email text is converted to normalized TF-IDF vectors and classified using multinomial Naive Bayes. A local browser interface accepts a subject and message, displays a predicted class and model score, and explains influential terms. A training script validates labeled CSV data, removes exact normalized duplicates, creates a reproducible stratified holdout, and saves evaluation results and a JSON model. The included dataset contains 50 manually authored synthetic messages. The measured demonstration results illustrate the workflow; they do not establish performance on real email.

## 1. Problem statement and objectives

Unwanted email creates distraction and can carry deceptive offers. This project studies whether word patterns can distinguish labeled spam from legitimate messages. Its objectives are to implement a complete text classification pipeline, demonstrate a usable interface, measure classification errors, and document reproducible experiments. It does not attempt automatic inbox management or malware detection.

## 2. Requirements

Python 3.10 or later, a modern browser, and a terminal are sufficient. No external Python libraries or remote services are required. Functional requirements include CSV import, training, text classification, term contribution display, evaluation, and model persistence. Input validation, local processing, and reproducible splits support reliability and privacy.

## 3. Architecture

```text
Labeled CSV → validation and deduplication → stratified train/test split
    Training texts → vocabulary + IDF → TF-IDF vectors → Naive Bayes
    Test texts     → frozen transformation → predictions → evaluation JSON
    Evaluated model → model JSON → local HTTP server → browser prediction
```

The browser submits text to POST /api/predict. The server uses a model loaded at startup and returns JSON. GET /api/metrics provides the evaluation metadata. Email requests are neither stored nor logged by the application. The server binds to the loopback address.

## 4. Dataset and preparation

The included demo.csv has 25 ham and 25 spam examples authored for this project. Messages contain simple workplace or personal correspondence and obvious promotional or prize language. This artificial separation makes the task easier than real email classification.

The CSV schema is text,label, with labels ham and spam. Missing text, unsupported labels, and conflicting labels for the same normalized text are rejected. Identical token sequences are deduplicated before splitting. Each class must contain at least five distinct messages. The script sorts each class and applies a seeded shuffle with seed 42; approximately 20% of each class is reserved for testing. For the demo this produces 40 training and 10 test messages.

## 5. Algorithm

1. Lowercase text and extract English alphanumeric tokens with a regular expression.
2. Learn the training vocabulary and document frequencies. For term t, use IDF(t) = ln((1 + N)/(1 + df(t))) + 1.
3. Multiply raw term counts by IDF, then normalize each document vector to unit Euclidean length. Ignore unseen terms.
4. Accumulate normalized feature values by class. Apply Laplace smoothing with alpha = 1: P(t|c) = (class feature total for t + 1)/(sum of class feature totals + vocabulary size).
5. Compute each class score as log P(c) + sum over terms of TF-IDF(t) × log P(t|c).
6. Convert the two scores to a spam score using a stable logistic calculation. Select spam at a score of at least 0.5.

The fractional TF-IDF formulation is a practical Naive Bayes approach rather than a literal model of integer token events. Its conditional independence assumption ignores word order and dependencies. The score is uncalibrated. A term's displayed contribution is its TF-IDF value multiplied by the difference between spam and ham log likelihoods; this explains the model arithmetic, not a causal reason why an email is spam.

Training time and model storage are modest for educational text datasets. The implementation stores a likelihood for every vocabulary item in both classes; very large vocabularies would benefit from pruning and established sparse-matrix libraries.

## 6. Experiment and measured results

Run python train.py to reproduce the included experiment. Dataset SHA-256: 90bca324f6a63d70fafb00327501428c4053e5c23f1022b4938ccef5ba3a54b7. The vocabulary and IDF are fitted only on the 40 training messages. The saved model is the evaluated training-only model; the test messages are not used to refit it.

| Metric | Measured demo result |
|---|---:|
| Accuracy | 100% (10/10) |
| Spam precision | 1.000 |
| Spam recall | 1.000 |
| Spam F1 | 1.000 |

| Actual / predicted | Ham | Spam |
|---|---:|---:|
| Ham | 5 | 0 |
| Spam | 0 | 5 |

Precision measures the fraction of predicted spam that is labeled spam. Recall measures the fraction of labeled spam detected. F1 is their harmonic mean. Accuracy measures the fraction of all correct predictions. A false positive incorrectly flags legitimate email; a false negative misses spam.

These results concern only ten synthetic test messages. They are not evidence of 100% real-world accuracy. The tiny balanced dataset has obvious vocabulary patterns and does not represent realistic class prevalence, multilingual text, sophisticated phishing, or changing campaigns. No statistically meaningful production claim follows from this experiment.

## 7. Verification

The included unittest suite verifies deterministic disjoint splitting, exclusion of unseen terms from the learned vocabulary, expected prediction direction on a controlled example, JSON serialization equivalence, empty input rejection, and metric bounds. Run python -m unittest discover -s tests -v. HTTP handlers were also checked using an in-memory connection for valid predictions, blank text, invalid input types, and unknown routes. Live loopback requests were blocked by the execution environment's socket permissions, so live browser behavior and visual layout have not been verified here.

## 8. Limitations and ethical considerations

The system does not parse MIME email, extract attachments, resolve links, inspect sender authentication, or evaluate sender reputation. HTML and signatures are treated as ordinary text. Non-English scripts are unsupported. Prior class frequency and vocabulary can bias results; unfamiliar text can receive a score driven largely by priors. The interface flags emails with no recognized terms as insufficient vocabulary.

Training data must be collected with appropriate permission and stripped of unnecessary personal information. Never publish private inbox data or committed credentials. False positives can conceal important correspondence, while false negatives can expose users to scams. This prototype should support learning and manual review, not automated deletion or safety certification. The local development server should not be publicly exposed.

## 9. Future work

Collect a diverse, consented email corpus and document its provenance and license. Separate test data by time and campaign or thread to reduce leakage. Compare this baseline against logistic regression on identical splits. Evaluate precision-recall curves, uncertainty, calibration, class imbalance, and per-category failure modes. Add MIME/HTML extraction, character features, multilingual support, and user feedback only after a trustworthy evaluation workflow is established.

## 10. Conclusion

The delivered project demonstrates the full path from labeled text to a browser-based prediction using reproducible training and transparent evaluation. Its measured result validates the demonstration workflow on simple synthetic examples. A separate real-email study is required before claiming useful operational performance.

## References

1. scikit-learn. Text feature extraction. https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
2. scikit-learn. MultinomialNB. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
3. Python documentation. http.server. https://docs.python.org/3/library/http.server.html

## Appendix: demonstration and viva preparation

Demonstration sequence: explain the CSV schema; run training; discuss the confusion matrix and synthetic-data limitation; start the app; enter a workplace message and a prize message; try unfamiliar text; run tests; explain the saved model.

- Why TF-IDF? It reduces the relative influence of terms that occur across many training documents.
- Why Naive Bayes? It is a simple, efficient text-classification baseline with interpretable token weights.
- What prevents vocabulary leakage? Splitting occurs before vocabulary and IDF fitting.
- Why report precision and recall? Accuracy alone can hide missed spam or excessive false alarms, especially with imbalanced classes.
- Is the displayed score confidence? It is an uncalibrated model score and may be misleading on unfamiliar text.
- Why is perfect demo accuracy insufficient? The test set is tiny, synthetic, and unrepresentative of real email.
- How would you improve evaluation? Use a larger independent corpus, chronological and campaign-aware splits, and untouched final test data.
