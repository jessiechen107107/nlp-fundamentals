#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# a
import os

def read_data(pos_train_path, neg_train_path, pos_test_path, neg_test_path):
    def load_folder(folder_path, label):
        texts = []
        labels = []
        # Sort filenames in ascending order to ensure consistent ordering
        filenames = sorted(os.listdir(folder_path))
        for fname in filenames:
            if fname.endswith('.txt'):
                fpath = os.path.join(folder_path, fname)
                # Use errors='replace' to handle any unusual encodings in the dataset
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    texts.append(f.read())
                labels.append(label)
        return texts, labels

    # Load positive and negative training data
    pos_train_texts, pos_train_labels = load_folder(pos_train_path, label=1)
    neg_train_texts, neg_train_labels = load_folder(neg_train_path, label=0)

    # Load positive and negative test data
    pos_test_texts,  pos_test_labels  = load_folder(pos_test_path,  label=1)
    neg_test_texts,  neg_test_labels  = load_folder(neg_test_path,  label=0)

    # Merge positive and negative samples for train and test
    train_texts  = pos_train_texts  + neg_train_texts
    train_labels = pos_train_labels + neg_train_labels

    test_texts  = pos_test_texts  + neg_test_texts
    test_labels = pos_test_labels + neg_test_labels

    return train_texts, train_labels, test_texts, test_labels

train_texts, train_labels, test_texts, test_labels = read_data(
    pos_train_path='pos_train',
    neg_train_path='neg_train',
    pos_test_path='pos_test',
    neg_test_path='neg_test'
)


# In[3]:


# b
from sklearn.model_selection import train_test_split
train_texts, dev_texts, train_labels, dev_labels = train_test_split(
    train_texts,
    train_labels,
    test_size=200/1800,
    stratify=train_labels,
    random_state=42
)


# In[4]:


# c
from transformers import DistilBertTokenizerFast

# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize train, dev, and test texts
# truncation=True  : truncate sequences longer than max_length (512 tokens)
# padding=True     : pad shorter sequences to the same length within each batch
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
dev_encodings   = tokenizer(dev_texts,   truncation=True, padding=True)
test_encodings  = tokenizer(test_texts,  truncation=True, padding=True)


# In[5]:


# d
import torch

class PolarityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        # Store the tokenized encodings and labels
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Retrieve a single sample by index as a dictionary of tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for this sample
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.labels)

# Create PolarityDataset objects for train, dev, and test
train_dataset = PolarityDataset(train_encodings, train_labels)
dev_dataset   = PolarityDataset(dev_encodings,   dev_labels)
test_dataset  = PolarityDataset(test_encodings,  test_labels)


# In[9]:


# e
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Load the pre-trained DistilBert model for sequence classification
# num_labels=2 because we have two classes: pos and neg
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)

# Use GPU if available, otherwise use CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Create DataLoaders for train, dev, and test datasets
# batch_size=16 as required in part (h)
# shuffle=True for training to prevent the model from learning the order of samples
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader   = DataLoader(dev_dataset,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

# Initialize the AdamW optimizer with learning rate 5e-5 as required in part (h)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Set the number of epochs
num_epochs = 3

# Training loop
for epoch in range(num_epochs):
    # --- Training phase ---
    model.train()  # Set model to training mode
    total_train_loss = 0

    for batch in train_loader:
        # Move each tensor in the batch to the correct device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass: compute model output and loss
        outputs = model(**batch)
        loss = outputs.loss

        # Accumulate training loss
        total_train_loss += loss.item()

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Clear gradients for the next step
        optimizer.zero_grad()

    # --- Validation phase ---
    model.eval()  # Set model to evaluation mode
    total_dev_loss = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_dev_loss += outputs.loss.item()

    # Calculate average losses
    avg_train_loss = total_train_loss / len(train_loader)
    avg_dev_loss   = total_dev_loss   / len(dev_loader)

    # f
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_dev_loss:.4f}")


# In[10]:


# g
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set model to evaluation mode
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():  # Disable gradient computation for inference
    for batch in test_loader:
        # Move batch to the correct device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass: get model outputs
        outputs = model(**batch)

        # Get predicted class by taking the argmax of the logits
        preds = torch.argmax(outputs.logits, dim=1)

        # Collect predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

# Calculate metrics (pos=1 is the positive class)
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, pos_label=1)
recall    = recall_score(all_labels, all_preds,    pos_label=1)
f1        = f1_score(all_labels, all_preds,        pos_label=1)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

