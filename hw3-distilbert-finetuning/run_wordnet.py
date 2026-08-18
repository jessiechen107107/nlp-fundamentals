# a
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn

# Get the first synset of "house"
house_synset = wn.synsets('house')[0]
print(f"First synset of 'house': {house_synset}")
print(f"Definition: {house_synset.definition()}")

# Find hypernyms
hypernyms = house_synset.hypernyms()
print(f"\nHypernyms of '{house_synset.name()}':")
for h in hypernyms:
    print(f"  - {h.name()}: {h.definition()}")

# b
# Get the first synset of each word
mouse_synset = wn.synsets('mouse')[0]
horse_synset = wn.synsets('horse')[0]
vacation_synset = wn.synsets('vacation')[0]

print(f"mouse  -> {mouse_synset.name()}: {mouse_synset.definition()}")
print(f"horse  -> {horse_synset.name()}: {horse_synset.definition()}")
print(f"vacation -> {vacation_synset.name()}: {vacation_synset.definition()}")

# Compute path similarity
sim_mouse_horse = mouse_synset.path_similarity(horse_synset)
sim_horse_vacation = horse_synset.path_similarity(vacation_synset)

print(f"\nPath similarity (mouse, horse):    {sim_mouse_horse}")
print(f"Path similarity (horse, vacation): {sim_horse_vacation}")

# Compare
if sim_mouse_horse > sim_horse_vacation:
    print("\n=> (mouse, horse) is more similar")
else:
    print("\n=> (horse, vacation) is more similar")

# c
import numpy as np

def load_glove(filepath):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def cosine_similarity(v1, v2):
    # Compute dot product
    dot_product = np.dot(v1, v2)
    # Compute magnitudes
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_product / magnitude

# Load GloVe vectors
glove = load_glove('glove.6B.50d.txt')

# Compute cosine similarities
sim_mouse_horse_glove = cosine_similarity(glove['mouse'], glove['horse'])
sim_horse_vacation_glove = cosine_similarity(glove['horse'], glove['vacation'])

print(f"\nCosine similarity (mouse, horse):    {sim_mouse_horse_glove:.4f}")
print(f"Cosine similarity (horse, vacation): {sim_horse_vacation_glove:.4f}")

if sim_mouse_horse_glove > sim_horse_vacation_glove:
    print("=> (mouse, horse) is more similar")
else:
    print("=> (horse, vacation) is more similar")