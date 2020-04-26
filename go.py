import random
import collections

from data import *
from lib import *
from classifiers import NaiveBayesClassifier

random.seed(0)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, 'data')

data = get_data(DATA_DIR)
training_set, testing_set = split_data(data, .75)

# print(count_words(training_set))

classifier = NaiveBayesClassifier(k=.0005)
classifier.train(training_set)
cars = classifier.word_probs[:10]

classified = [(subject, is_spam, classifier.classify(subject)) for subject, is_spam, in testing_set]
counts = collections.Counter((is_spam, spam_probability > .5) for _, is_spam, spam_probability in classified)

print('(is_spam, classified_as_spam')
print('--------------------------------')

for x in counts:
    print(x, counts[x])
