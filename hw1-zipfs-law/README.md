# HW1 — Corpus Statistics

**Author:** Yuan Yin Chen (solo assignment)

## What is this?

Two corpus-statistics exercises: (1) checking whether real dialogue text follows Zipf's law, and
(2) building a bigram language model with smoothing and using it to score sentences.

## What problem does it solve?

Zipf's law and n-gram smoothing are foundational because they explain *why* NLP systems need
smoothing at all: raw n-gram counts assign zero probability to any unseen sequence, which breaks
scoring/generation on real text. This exercise makes both problems and the fix concrete on real
data instead of a toy example.

## How it works

- `hw1-2.py` — parses the [Diplomacy](https://sites.google.com/view/qanta/projects/diplomacy)
  in-game dialogue dataset (`train.jsonl`, not included) with NLTK's `sent_tokenize`/
  `word_tokenize`, builds a word-frequency table, and plots frequency vs. rank on a log-log scale.
- `hw1-4.py` — builds a bigram language model over two Brown corpus categories (News, Romance),
  computes sentence log-probability with and without add-one (Laplace) smoothing.

## Tech stack

Python, NLTK (`sent_tokenize`, `word_tokenize`, Brown corpus), matplotlib.

## Results

- The Diplomacy corpus contains **17,901 sentences** and **321,214 word tokens** (8,780 unique
  types after lowercasing).
- The most frequent word type is **"i"**; the 5th most frequent is **"you"**.
- The log-log rank/frequency plot (`zipf_law_graph.png`) is a straight line with negative slope,
  confirming Zipf's law (f(r) ∝ 1/r) — a small set of function words dominate, with a long tail of
  rare words.
- Without smoothing, the bigram probability of *"\<s\> I loved her when she laughed \</s\>"* under
  the **News** model is **0** (an unseen bigram makes the whole sentence probability collapse to
  zero), while the **Romance** model gives log P = **-27.41** for the same sentence — a direct
  demonstration of the sparse-data problem raw n-gram models have.
- After add-one smoothing: News model log P = **-61.15**, Romance model log P = **-49.48** — both
  now finite and comparable, at the cost of redistributing probability mass away from seen events.

## What I learned

Seeing the News model's unsmoothed probability hit exactly zero — not "very small," but
mathematically zero — made the sparse-data problem concrete in a way that reading about smoothing
in the abstract didn't. It also showed why the *choice* of training corpus matters as much as the
smoothing method: the Romance-domain bigrams happened to cover this sentence better than the
News-domain ones, so the two "same" bigram models gave meaningfully different scores for identical
text.

## How to run it

```bash
pip install nltk matplotlib
python hw1-2.py   # requires train.jsonl (Diplomacy dataset) in the same folder
python hw1-4.py   # downloads the Brown corpus via nltk.download('brown') on first run
```

Full write-up: [`report.pdf`](./report.pdf).
