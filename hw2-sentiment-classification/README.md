# HW2 — Classical Sequence Models

**Author:** Yuan Yin Chen (solo assignment)

## What is this?

Two classical NLP algorithms implemented and evaluated: a logistic regression sentiment classifier,
and the Viterbi algorithm for HMM part-of-speech tagging, built from scratch (no HMM library
allowed).

## What problem does it solve?

Before reaching for a transformer, it's worth knowing what a simple, interpretable model can and
can't do on the same task — and for POS tagging, worth actually implementing the classic
dynamic-programming decoding algorithm instead of only calling a library, to understand why the
DP formulation (rather than brute-force enumeration) is what makes tagging tractable.

## How it works

- `run_lr.py` — sentiment classifier: spaCy for tokenization/feature extraction (adjectives only),
  scikit-learn logistic regression, evaluated with F1 on the Pang & Lee movie-review polarity
  corpus (`pos_train`/`neg_train`/`pos_test`/`neg_test`, not included — standard academic dataset).
- `run_viterbi.py` — from-scratch implementation of the Viterbi algorithm for HMM POS tagging:
  builds delta/backpointer matrices from initial/transition/emission probabilities, decodes the
  most-likely tag sequence, and reconstructs it via backtracking.

## Tech stack

Python, spaCy, scikit-learn (`LogisticRegression`, `f1_score`), NumPy.

## Results

- **Logistic regression:** training F1 = **1.0**, test F1 = **0.78** — the visible gap indicates
  overfitting on the training-set-specific adjective vocabulary.
- **Viterbi:** the algorithm's output matched a hand-computed HMM decode exactly for the test
  sentence "will Cherry spot Patrick" (most likely tag sequence VNVN with the same per-state
  probabilities), confirming the implementation against ground truth rather than just "runs without
  error."

## What I learned

- Adjective-only features are limited because they ignore negation ("not good" reads the same as
  "good") and drop sentiment-bearing verbs and nouns ("love"/"hate", "masterpiece"/"disaster") —
  the report's own discussion of the 1.0-vs-0.78 F1 gap. That's a concrete case of a model
  overfitting not because it's too complex, but because its *feature space* is too narrow to
  generalize, which is a different failure mode than "add more regularization."
- Implementing Viterbi from scratch (rather than using an HMM library) and cross-checking it
  against hand-computed probabilities was the useful part — it's easy to get a DP table
  implementation that runs and produces *a* plausible-looking answer without actually being
  correct, and manual verification is what catches that.

## How to run it

```bash
pip install spacy scikit-learn numpy
python -m spacy download en_core_web_sm
python run_lr.py         # expects pos_train/neg_train/pos_test/neg_test folders
python run_viterbi.py
```

Full write-up: [`report.pdf`](./report.pdf).
