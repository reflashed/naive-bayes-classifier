from lib import *

class NaiveBayesClassifier:
    def __init__(self, k=.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):
        # count spam and non-spam msgs
        num_spams = len([is_spam for msg, is_spam in training_set if is_spam])
        num_non_spams = len(training_set) - num_spams

        # run training data through our 'pipeline'
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts, num_spams, num_non_spams, self.k)

    def classify(self, msg):
        return spam_probability(self.word_probs, msg)
