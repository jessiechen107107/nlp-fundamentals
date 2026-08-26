# HW3 — Transformer Fine-Tuning & Lexical Semantics

**Author:** Yuan Yin Chen (solo assignment)

## What is this?

Fine-tuning DistilBERT for sentiment classification on the same movie-review corpus as HW2 (to
directly compare against the classical baseline), plus a WordNet exploration of lexical relations.

## What problem does it solve?

HW2 established a classical baseline (0.78 test F1) and showed it overfits on a narrow feature set.
This assignment tests whether a pretrained transformer actually fixes that — and, through four
ablations, isolates *which part* of "using a transformer" is responsible for any improvement
(the architecture? full fine-tuning specifically? hyperparameters?), rather than treating "used
BERT" as a single unexamined lever.

## How it works

- `run_distillbert.py` (+ `_i1`/`_i2`/`_i3`/`_j` variants for each experiment below) — fine-tunes
  `distilbert-base-uncased` with a classification head on the HW2 movie-review data.
- `run_wordnet.py` — explores WordNet lexical relations via NLTK: synsets, definitions, hypernyms,
  and semantic similarity between word pairs.

## Tech stack

Python, HuggingFace `transformers` (DistilBERT), PyTorch, NLTK (WordNet).

## Results

Baseline (AdamW, lr=5e-5, batch size 16, 3 epochs): **test accuracy 0.885, precision 0.943, recall
0.820, F1 0.877** — a ~10-point F1 jump over HW2's logistic regression baseline (0.78).

Four ablations, each changing one variable from the baseline:

| Setting | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Baseline** (full fine-tune) | 0.8850 | 0.9425 | 0.8200 | 0.8770 |
| Higher LR (1e-4) | 0.5000 | — | — | 0.6667 |
| Smaller batch (8) | 0.8500 | — | 0.8900 | 0.8558 |
| More epochs (5) | 0.8350 | — | — | 0.8520 |
| Layer freezing (head only) | 0.7450 | 0.7816 | 0.6800 | 0.7273 |

- **Higher learning rate (1e-4) broke training entirely** — the model collapsed to predicting every
  sample positive (50% accuracy, chance level).
- **Smaller batch size (8)** traded precision for recall (0.82→0.89 recall) at a small accuracy
  cost — noisier gradients made the model more liberal about predicting positive.
- **More epochs (5)** caused clear overfitting: validation loss rose from 0.44 to 0.82 while
  training loss kept falling, and test accuracy dropped to 0.835.
- **Freezing the transformer body** (training only the classification head) dropped F1 to 0.7273 —
  *below* the HW2 logistic regression baseline (0.78) — showing that DistilBERT's pretrained
  features alone, without task-adaptation, aren't enough; full fine-tuning is what earned the
  improvement over the classical model.

## What I learned

- The layer-freezing ablation was the most informative result in the assignment: it directly
  disproves the naive assumption "a transformer will beat a classical model just by being a
  transformer." Frozen DistilBERT actually *underperformed* the HW2 logistic regression baseline —
  the gain came specifically from full fine-tuning adapting the pretrained representations to this
  task, not from the architecture alone.
- The learning-rate ablation was a fast, cheap way to see instability firsthand: 5e-5 trains fine,
  2x that (1e-4) collapses to a degenerate all-positive predictor. That's now my default intuition
  for why fine-tuning guides are conservative about learning rate for pretrained models.
- Comparing precision/recall (not just accuracy) across the ablations showed that "worse" isn't
  always uniform — the smaller-batch run was actually *better* on recall than the baseline despite
  lower accuracy, which is a distinction a single accuracy number would have hidden.

## How to run it

```bash
pip install transformers torch nltk
python run_distillbert.py     # baseline config
python run_distillbert_i1.py  # higher learning rate
python run_distillbert_i2.py  # smaller batch size
python run_distillbert_i3.py  # more epochs
python run_distillbert_j.py   # frozen transformer layers
python run_wordnet.py
```

Full write-up: [`report.pdf`](./report.pdf).
