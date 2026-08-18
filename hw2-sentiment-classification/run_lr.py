import os
import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def load_documents(folder_path, label):
    docs, labels = [], []
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                docs.append(f.read())
            labels.append(label)
    return docs, labels


pos_train_docs, pos_train_labels = load_documents('pos_train', label=1)
neg_train_docs, neg_train_labels = load_documents('neg_train', label=0)
pos_test_docs,  pos_test_labels  = load_documents('pos_test',  label=1)
neg_test_docs,  neg_test_labels  = load_documents('neg_test',  label=0)

train_docs   = pos_train_docs + neg_train_docs
train_labels = pos_train_labels + neg_train_labels
test_docs    = pos_test_docs + neg_test_docs
test_labels  = pos_test_labels + neg_test_labels

# a
nlp = spacy.load('en_core_web_sm')

def get_adjectives(text, nlp_model):
    doc = nlp_model(text)
    return {token.text.lower() for token in doc if token.pos_ == 'ADJ'}

adj_vocab = set()
for text in train_docs:
    adj_vocab.update(get_adjectives(text, nlp))

adj_vocab = sorted(adj_vocab)
adj_index = {adj: i for i, adj in enumerate(adj_vocab)}
vocab_size = len(adj_vocab)

# b
def vectorize(docs, nlp_model, adj_index, vocab_size):
    X = np.zeros((len(docs), vocab_size), dtype=np.int8)
    for i, text in enumerate(docs):
        for adj in get_adjectives(text, nlp_model):
            if adj in adj_index:
                X[i, adj_index[adj]] = 1
    return X

X_train = vectorize(train_docs, nlp, adj_index, vocab_size)
y_train = np.array(train_labels)
X_test  = vectorize(test_docs, nlp, adj_index, vocab_size)
y_test  = np.array(test_labels)

# c
clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
clf.fit(X_train, y_train)

# d
train_f1 = f1_score(y_train, clf.predict(X_train))
test_f1  = f1_score(y_test,  clf.predict(X_test))

print(f"(e) F1 Score on TRAINING data: {train_f1:.4f}")
print(f"(f) F1 Score on TEST     data: {test_f1:.4f}")