import sys
import os
import math

spam_dir = set()
ham_dir = set()
spam_word_freq = dict()
ham_word_freq = dict()
total_words_spam = 0
total_words_ham = 0
vocab = set()


def read_files(train_path):
    count = 0
    for root, dirs, files in os.walk(train_path):
        if len(files)>0:
            if root.split('/')[-1] == 'spam':
                #print(root, len(files),files[0],  'spam')
                for file_name in files:
                    spam_dir.add(root+'/'+file_name)
            else:
                #print(root, len(files), 'ham')
                for file_name in files:
                    ham_dir.add(root+'/'+file_name)

    return spam_dir, ham_dir


def word_counts():
    global total_words_spam
    global total_words_ham

    for spam_file in spam_dir:
        document = open(spam_file, "r", encoding="latin1")
        for line in document:
            line = line.split()
            for word in line:
                word = word.lower()
                # punc is not removed - >, --, *, (, )
                # handling numbers
                if word.isnumeric():
                    word = "%NUMBER%"
                if word in spam_word_freq:
                    spam_word_freq[word] +=1
                else:
                    spam_word_freq[word] = 1
                total_words_spam +=1
                vocab.add(word)

    for ham_file in ham_dir:
        document_ham = open(ham_file, "r", encoding = "latin1")
        for line_ham in document_ham:
            line_ham = line_ham.split()
            for word_ham in line_ham:
                # change made, look
                word_ham = word_ham.lower()
                # punc is not removed - >, --, *, (, ), numbers
                # handling numbers
                if word_ham.isnumeric():
                    word_ham = "%NUMBER%"
                if word_ham in ham_word_freq:
                    ham_word_freq[word_ham] +=1
                else:
                    ham_word_freq[word_ham] = 1
                total_words_ham +=1
                vocab.add(word_ham)


def extract_prob():
    spam_word_prob = dict()
    ham_word_prob = dict()
    prob_spam = dict()
    prob_ham = dict()

    total_documents = len(ham_dir)+len(spam_dir)
    prob_spam['value'] = math.log(len(spam_dir)/total_documents)
    prob_ham['value'] = math.log(len(ham_dir)/total_documents)

    for word in spam_word_freq:
        prob_value = math.log((spam_word_freq[word] +1)/(total_words_spam + len(vocab)))
        spam_word_prob[word] = prob_value
    for word2 in ham_word_freq:
        prob_value2 = math.log((ham_word_freq[word2] +1)/(total_words_ham + len(vocab)))
        ham_word_prob[word2] = prob_value2

    ## extra info
    info = dict()
    info['tot_spam'] = total_words_spam
    info['tot_ham'] = total_words_ham
    info['len_vocab'] = len(vocab)

    return prob_spam, prob_ham, spam_word_prob, ham_word_prob, info


def write_model(model_filename, prob_array):
    model_file = open(model_filename, "w", encoding="latin1")
    model_file.write(str(prob_array))
    model_file.close()


if __name__ == "__main__":
    #train_path = './train/'
    train_path = sys.argv[1]
    model_filename = 'nbmodel.txt'
    read_files(train_path)
    word_counts()
    prob_spam, prob_ham, spam_word_prob, ham_word_prob, info = extract_prob()
    prob_array = [prob_spam, prob_ham, spam_word_prob, ham_word_prob, info]
    write_model(model_filename, prob_array)
