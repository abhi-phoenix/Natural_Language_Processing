import sys
import os
import math
import ast
import random

test_dir = set()

def read_files(train_path):    
    count = 0 
    for root, dirs, files in os.walk(train_path):
        if len(files)>0:
            ## next line is required or not? test in vocareum
            #files.remove('.DS_Store')
            for file_name in files:
                test_dir.add(root+'/'+file_name)


def read_model(model_filename):
    model_file = open(model_filename, "r", encoding="latin1")
    lines = model_file.readlines()
    for line in lines:
        model_data = ast.literal_eval(line)

    return model_data
        
                    
def predict(model, output_file):

    output_file_pointer = open(output_file, "w", encoding="latin1")
    prob_spam = model[0]['value']
    prob_ham = model[1]['value']
    spam_word_prob = model[2]
    ham_word_prob = model[3]
    info = model[4]
    total_words_spam = info['tot_spam']
    total_words_ham = info['tot_ham']
    vocab_length = info['len_vocab']

    
    for file in test_dir:
        document = open(file, "r", encoding="latin1")

        spam_mess_probability = prob_spam
        ham_mess_probability = prob_ham
        for line in document:

            line = line.split()

            
            for word in line:

                word = word.lower()

                # punc is not removed - >, --, *, (, )

                # handling numbers

                if word.isnumeric():

                    word = "%NUMBER%sbdfjksd"

                if word not in spam_word_prob and ham_word_prob:
                    continue
                else:
                        # if one of the words are not present either in spam or in ham ???, need vocabulary value and number of words in class ck value
                    if word not in spam_word_prob:
                        val = math.log(1/(total_words_spam + vocab_length))
                        spam_mess_probability += val
                        ham_mess_probability += ham_word_prob[word]
                    elif word not in ham_word_prob:
                        val = math.log(1/(total_words_ham + vocab_length))
                        spam_mess_probability += spam_word_prob[word]
                        ham_mess_probability += val
                    else:
                        spam_mess_probability += spam_word_prob[word]
                        ham_mess_probability += ham_word_prob[word]
                    

        if spam_mess_probability > ham_mess_probability:
            label = "spam"
        elif spam_mess_probability == ham_mess_probability:
            random_val = random.randint(0,1)
            if random_val == 0:
                label = "spam"
            else:
                label = "ham"
        else:
            label = "ham"
                    
        # check this line
        output_file_pointer.write('%s\t%s\n' % (label,file))
                                  

if __name__ == "__main__":
    # / inlcuded or not? - check in vocareum about paths returned
    #test_path = 'dev'
    test_path = sys.argv[1]
    output_file = 'nboutput.txt'
    model_filename = 'nbmodel.txt'
    
    read_files(test_path)
    model = read_model(model_filename)
    predict(model, output_file)
    
    
    

    


