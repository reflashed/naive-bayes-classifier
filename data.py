import random 
import os, re

def get_data(data_dir):
    easy_ham_dir = os.path.join(data_dir, 'easy_ham')
    hard_ham_dir = os.path.join(data_dir, 'hard_ham')
    spam_dir = os.path.join(data_dir, 'spam')

    easy_ham_msgs = get_msgs(easy_ham_dir, is_spam=False)
    hard_ham_msgs = get_msgs(hard_ham_dir, is_spam=False)
    spam_msgs = get_msgs(spam_dir, is_spam=True)

    return easy_ham_msgs + hard_ham_msgs + spam_msgs

def get_files_abs(d):
    return [os.path.join(d, x) for x in os.listdir(d)]

def read_msg(msg, is_spam):
    subject = False
    for line in msg.split('\n'):
        if line.startswith("Subject:"):
            subject = re.sub(r"^Subject: ", "", line).strip()

    if subject == False:
        return False

    return (subject, is_spam)

def get_msgs(d, is_spam):
    msgs = []
    for msg_path in get_files_abs(d):
        with open(msg_path, 'r', encoding='ISO-8859-1') as f:
            msg = f.read()

        parsed_msg = read_msg(msg, is_spam)
        if parsed_msg:
            msgs.append(parsed_msg)

    return msgs

def split_data(data, frac):
    train, test = [], []

    frac_percent = frac*100

    while len(data) > 0:
        datum = random.choice(data)
        data.remove(datum)

        if random.randint(1, 100) <= frac_percent:
            train.append(datum)
        else:
            test.append(datum)

    return (train, test)
