# HW2 — Classical Sequence Models

- `run_lr.py` — sentiment classifier: spaCy for tokenization/feature extraction, scikit-learn
  logistic regression, evaluated with F1 on a held-out movie-review test split
  (`pos_train`/`neg_train`/`pos_test`/`neg_test`, the Pang & Lee movie-review polarity corpus —
  not included, standard academic dataset).
- `run_viterbi.py` — from-scratch implementation of the Viterbi algorithm for HMM POS tagging
  (initial/transition/emission probabilities, dynamic-programming decode with backpointers).

Full write-up: [`report.pdf`](./report.pdf).
