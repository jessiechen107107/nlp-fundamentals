# NLP Fundamentals

Three assignments from CSCI-B/P NLP, Spring 2026 (Indiana University Indianapolis), covering
corpus statistics, classical sequence models, and transformer fine-tuning.

| # | Topic | Techniques |
|---|---|---|
| [1](./hw1-zipfs-law) | Corpus statistics | Zipf's law over a dialogue corpus; n-gram language modeling with perplexity on the Brown corpus |
| [2](./hw2-sentiment-classification) | Classical sequence models | Logistic regression sentiment classifier (spaCy + scikit-learn); HMM POS tagging via the Viterbi algorithm |
| [3](./hw3-distilbert-finetuning) | Transformer fine-tuning | Fine-tuning DistilBERT for sentiment classification; WordNet lexical relations (hypernyms, similarity) |

Each folder has the assignment's write-up (`report.pdf`) and code. Datasets (movie-review corpora,
Brown corpus, GloVe embeddings) are not included — they're standard NLTK/course-provided corpora,
downloaded automatically by the scripts (`nltk.download(...)`) or documented in each report.
