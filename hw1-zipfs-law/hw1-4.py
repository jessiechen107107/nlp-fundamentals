import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import math

# Download Brown corpus
nltk.download('brown')

def preprocess_sentences(sentences):
    """
    Preprocess sentences: lowercase, remove punctuation-only tokens, add <s> and </s>
    """
    processed = []
    
    for sent in sentences:
        # Lowercase all tokens
        lowercased = [word.lower() for word in sent]
        
        # Remove punctuation-only tokens
        # Keep tokens that have at least one alphanumeric character
        filtered = [token for token in lowercased 
                   if any(c.isalnum() for c in token)]
        
        # Add sentence start and end markers
        filtered = ['<s>'] + filtered + ['</s>']
        
        processed.append(filtered)
    
    return processed

def build_unigram_model(processed_sentences):
    """
    Build unigram model: count each word occurrence
    """
    unigram_counts = Counter()
    
    for sent in processed_sentences:
        for word in sent:
            unigram_counts[word] += 1
    
    return unigram_counts

def build_bigram_model(processed_sentences):
    """
    Build bigram model: count consecutive word pairs
    """
    bigram_counts = Counter()
    
    for sent in processed_sentences:
        # Create bigrams from consecutive words
        for i in range(len(sent) - 1):
            bigram = (sent[i], sent[i+1])
            bigram_counts[bigram] += 1
    
    return bigram_counts

def calculate_unigram_probabilities(unigram_counts):
    """
    Calculate MLE probabilities for unigrams
    """
    total_words = sum(unigram_counts.values())
    unigram_probs = {}
    
    for word, count in unigram_counts.items():
        unigram_probs[word] = count / total_words
    
    return unigram_probs

def calculate_bigram_probabilities(bigram_counts, unigram_counts):
    """
    Calculate MLE probabilities for bigrams
    """
    bigram_probs = {}
    
    for (w1, w2), count in bigram_counts.items():
        # P(w2|w1) = C(w1,w2) / C(w1)
        bigram_probs[(w1, w2)] = count / unigram_counts[w1]
    
    return bigram_probs

def calculate_sentence_probability(sentence, bigram_probs, use_log=True):
    """
    Calculate probability of a sentence using bigram model
    """
    # Parse the sentence - it already has <s> and </s>
    words = sentence.strip().split()
    
    if use_log:
        log_prob = 0.0
        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            if bigram in bigram_probs:
                log_prob += math.log(bigram_probs[bigram])
            else:
                # Unseen bigram, assign very small probability
                return float('-inf')
        return log_prob
    else:
        prob = 1.0
        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            if bigram in bigram_probs:
                prob *= bigram_probs[bigram]
            else:
                return 0.0
        return prob

def calculate_bigram_probabilities_smoothed(bigram_counts, unigram_counts, vocab_size):
    """
    Calculate add-one smoothed probabilities for bigrams
    """
    bigram_probs_smoothed = {}
    
    # Get all unique words
    all_words = set(unigram_counts.keys())
    
    # Calculate smoothed probabilities for observed bigrams
    for (w1, w2), count in bigram_counts.items():
        bigram_probs_smoothed[(w1, w2)] = (count + 1) / (unigram_counts[w1] + vocab_size)
    
    # For sentence probability calculation, we may need unseen bigrams too
    # But we'll compute them on-demand
    
    return bigram_probs_smoothed, all_words

def calculate_sentence_probability_smoothed(sentence, bigram_counts, unigram_counts, vocab_size):
    """
    Calculate probability with add-one smoothing
    """
    # Parse the sentence - it already has <s> and </s>
    words = sentence.strip().split()
    
    log_prob = 0.0
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        bigram = (w1, w2)
        
        # Get count (0 if unseen)
        bigram_count = bigram_counts.get(bigram, 0)
        w1_count = unigram_counts.get(w1, 0)
        
        # Add-one smoothing
        prob = (bigram_count + 1) / (w1_count + vocab_size)
        log_prob += math.log(prob)
    
    return log_prob

def main():
    # Load data
    news_data = brown.sents(categories='news')
    romance_data = brown.sents(categories='romance')
    
    # Preprocess
    print("Preprocessing...")
    news_processed = preprocess_sentences(news_data)
    romance_processed = preprocess_sentences(romance_data)
    
    # Build models
    news_unigrams = build_unigram_model(news_processed)
    romance_unigrams = build_unigram_model(romance_processed)
    
    news_bigrams = build_bigram_model(news_processed)
    romance_bigrams = build_bigram_model(romance_processed)
    
    # (a) Non-zero unigrams
    print("\n(a) Non-zero unigrams:")
    print(f"News corpus: {len(news_unigrams)} unique unigrams")
    print(f"Romance corpus: {len(romance_unigrams)} unique unigrams")
    
    # (b) Non-zero bigrams
    print("\n(b) Non-zero bigrams:")
    print(f"News corpus: {len(news_bigrams)} unique bigrams")
    print(f"Romance corpus: {len(romance_bigrams)} unique bigrams")
    
    # Calculate probabilities
    news_unigram_probs = calculate_unigram_probabilities(news_unigrams)
    romance_unigram_probs = calculate_unigram_probabilities(romance_unigrams)
    
    # (c) Top 10 unigrams
    print("\n(c) Top 10 most common unigrams:")
    print("\nNews corpus:")
    for word, count in news_unigrams.most_common(10):
        prob = news_unigram_probs[word]
        print(f"{word}: count={count}, P(w)={prob:.6f}")
    
    print("\nRomance corpus:")
    for word, count in romance_unigrams.most_common(10):
        prob = romance_unigram_probs[word]
        print(f"{word}: count={count}, P(w)={prob:.6f}")
    
    # (d) Top 10 bigrams
    news_bigram_probs = calculate_bigram_probabilities(news_bigrams, news_unigrams)
    romance_bigram_probs = calculate_bigram_probabilities(romance_bigrams, romance_unigrams)
    
    print("\n(d) Top 10 most common bigrams:")
    print("\nNews corpus:")
    for bigram, count in news_bigrams.most_common(10):
        prob = news_bigram_probs[bigram]
        print(f"{bigram}: count={count}, P(w2|w1)={prob:.6f}")
    
    print("\nRomance corpus:")
    for bigram, count in romance_bigrams.most_common(10):
        prob = romance_bigram_probs[bigram]
        print(f"{bigram}: count={count}, P(w2|w1)={prob:.6f}")
    
    # (e) & (f) Sentence probability
    test_sentence = "<s> i loved her when she laughed </s>"
    
    print("\n(e) Probability using news bigram model:")
    prob_news = calculate_sentence_probability(test_sentence, news_bigram_probs, use_log=False)
    log_prob_news = calculate_sentence_probability(test_sentence, news_bigram_probs, use_log=True)
    print(f"P(sentence) = {prob_news}")
    print(f"log P(sentence) = {log_prob_news}")
    
    print("\n(f) Probability using romance bigram model:")
    prob_romance = calculate_sentence_probability(test_sentence, romance_bigram_probs, use_log=False)
    log_prob_romance = calculate_sentence_probability(test_sentence, romance_bigram_probs, use_log=True)
    print(f"P(sentence) = {prob_romance}")
    print(f"log P(sentence) = {log_prob_romance}")
    
    # (g) With add-one smoothing
    news_vocab_size = len(news_unigrams)
    romance_vocab_size = len(romance_unigrams)
    
    print("\n(g) Probability with add-one smoothing:")
    log_prob_news_smooth = calculate_sentence_probability_smoothed(
        test_sentence, news_bigrams, news_unigrams, news_vocab_size)
    log_prob_romance_smooth = calculate_sentence_probability_smoothed(
        test_sentence, romance_bigrams, romance_unigrams, romance_vocab_size)
    
    print(f"News model: log P(sentence) = {log_prob_news_smooth}")
    print(f"Romance model: log P(sentence) = {log_prob_romance_smooth}")

if __name__ == "__main__":
    main()