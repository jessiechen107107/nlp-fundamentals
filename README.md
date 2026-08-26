# NLP Fundamentals

Three assignments from CSCI-B/P NLP, Spring 2026 (Indiana University Indianapolis), solo work,
covering corpus statistics, classical sequence models, and transformer fine-tuning — each folder
compares approaches on the same underlying question: how much does model sophistication actually
buy you, and where does it stop paying off?

| # | Topic | Techniques | Headline result |
|---|---|---|---|
| [1](./hw1-zipfs-law) | Corpus statistics | Zipf's law over a dialogue corpus; bigram LM with add-one smoothing on Brown corpus | 321K tokens confirm Zipf's law (power-law fit); smoothing rescues zero-probability sentences |
| [2](./hw2-sentiment-classification) | Classical sequence models | Logistic regression sentiment classifier (spaCy + scikit-learn); HMM POS tagging via Viterbi | Train F1 1.0 vs. test F1 0.78 (overfitting); Viterbi output matched manual HMM calculation exactly |
| [3](./hw3-distilbert-finetuning) | Transformer fine-tuning | Fine-tuning DistilBERT on the same sentiment task; WordNet lexical relations | 88.5% test accuracy / 0.877 F1 — a 10-point F1 jump over the logistic regression baseline |

Each folder has the assignment's write-up (`report.pdf`) and code. Datasets (movie-review corpora,
Brown corpus, GloVe embeddings) are not included — they're standard NLTK/course-provided corpora,
downloaded automatically by the scripts (`nltk.download(...)`) or documented in each report.

## What I learned across all three

The clearest thread across HW2 and HW3 is what "better model" actually buys you on the *same*
sentiment classification task and dataset: logistic regression on adjective features hit 0.78 test
F1 with a visible train/test gap (overfitting on shallow features), while fine-tuned DistilBERT
reached 0.877 F1 — but only with full fine-tuning; freezing the transformer body and training just
the classification head (HW3 part j) dropped accuracy to 74.5%, below the classical baseline. That
told me the performance gain wasn't just "used a transformer," it specifically required letting the
pretrained representations adapt to the task, which is a more precise (and more useful) lesson than
"transformers are better."
