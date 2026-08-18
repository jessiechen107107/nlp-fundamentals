# Import necessary libraries
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Download NLTK data (only need to run once)
nltk.download('punkt')
nltk.download('punkt_tab')

# Read the JSONL file line by line
data = []
with open('train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            json_obj = json.loads(line)
            data.append(json_obj)

# Extract only the messages field
messages = []
for item in data:
    if 'messages' in item and item['messages']:
        # If messages is a list, join them; if string, use directly
        if isinstance(item['messages'], list):
            messages.append(' '.join(item['messages']))
        else:
            messages.append(item['messages'])

# Write messages to data.txt (one per line)
with open('data.txt', 'w', encoding='utf-8') as f:
    for message in messages:
        f.write(message + '\n')

# Read the data.txt file
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# a
# Split into sentences using NLTK
sentences = sent_tokenize(text)

# Remove empty sentences
sentences = [s.strip() for s in sentences if s.strip()]

num_sentences = len(sentences)
print(f"(a) Number of sentences: {num_sentences}")

# b
# Split each sentence by spaces
tokens_b = []
for sentence in sentences:
    tokens_b.extend(sentence.split(' '))

# Remove empty strings
tokens_b = [t for t in tokens_b if t]

num_tokens_b = len(tokens_b)
print(f"(b) Number of tokens using split(' '): {num_tokens_b}")

# c
# Tokenize using NLTK's word_tokenize
tokens_c = []
for sentence in sentences:
    tokens_c.extend(word_tokenize(sentence))

num_tokens_c = len(tokens_c)
print(f"(c) Number of tokens using word_tokenize(): {num_tokens_c}")

# d
# Lowercase all tokens
tokens_lower = [token.lower() for token in tokens_c]

num_tokens_d = len(tokens_lower)
num_types_d = len(set(tokens_lower))

print(f"(d) Number of tokens (after lowercase): {num_tokens_d}")
print(f"(d) Number of types (unique words): {num_types_d}")

# f
# Create frequency dictionary
word_freq = Counter(tokens_lower)

# Sort by frequency (descending)
sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

most_frequent = sorted_freq[0]
print(f"(f) Most frequent word: '{most_frequent[0]}' with frequency {most_frequent[1]}")

# g
fifth_frequent = sorted_freq[4]
print(f"(g) 5th most frequent word: '{fifth_frequent[0]}' with frequency {fifth_frequent[1]}")

# h
# Extract frequencies for plotting
ranks = list(range(1, len(sorted_freq) + 1))
frequencies = [freq for word, freq in sorted_freq]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(ranks, frequencies, 'b-', linewidth=2)
plt.xlabel('Rank (ranked words)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title("Zipf's Law: Word Rank vs Frequency (Diplomacy Dataset)", fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the graph
plt.savefig('zipf_law_graph.png', dpi=300, bbox_inches='tight')

plt.show()

