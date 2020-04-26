import re
import math
import collections

def tokenize(msg):
    msg = msg.lower()
    all_words = re.findall("[a-z0-9']+", msg)
    return set(all_words)

def count_words(training_set):
    '''training set consists of pairs (msg, is_spam)
    
    returns dict of words in which each key has a (spam_count, non_spam_count) tuple'''
    counts = collections.defaultdict(lambda: [0, 0])

    for msg, is_spam in training_set:
        for word in tokenize(msg):
            counts[word][0 if is_spam else 1] += 1

    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=.5):
    '''turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)'''
    return [(w,
            (spam + k) / (total_spams + 2 * k),
            (non_spam + k) / (total_non_spams + 2 * k))
            for w, (spam, non_spam) in counts.items()]

def spam_probability(word_probs, msg):
    msg_words = tokenize(msg)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    # iterate through each word in our covabulary
    for word, prob_if_spam, prob_if_not_spam in word_probs:
        # if 'word' appears in the msg
        if word in msg_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)
        # if 'word not in msg
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam += math.exp(log_prob_if_not_spam)

    return prob_if_spam / (prob_if_spam + prob_if_not_spam)
