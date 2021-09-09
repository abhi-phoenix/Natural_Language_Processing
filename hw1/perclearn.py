import sys
import os
import math
from collections import OrderedDict
import time

# orderedDict is slow
O_dict = OrderedDict()
#O_dict = dict()
# whole_dir = dict()
weights = dict()
avg_weights = dict()
bias = 0
avg_bias = 0
count = 1
max_iter = 100

"""

    O_dict = OrderedDict()
    O_dict['train2/spam/0002.2001-05-25.SA_and_HP.spam.txt'] = 1
    O_dict['train2/ham/0002.2001-05-25.SA_and_HP.ham.txt'] = -1
    O_dict['train2/spam/0002.2001-05-25.SA_and_HP.spam copy.txt'] = 1
    O_dict['train2/ham/0002.2001-05-25.SA_and_HP.ham copy.txt'] = -1
    O_dict['train2/ham/0002.2001-05-25.SA_and_HP.ham copy 2.txt'] = -1
    O_dict['train2/spam/0002.2001-05-25.SA_and_HP.spam copy 2.txt'] = 1
    O_dict['train2/ham/0002.2001-05-25.SA_and_HP.ham copy 3.txt'] = -1
    O_dict['train2/ham/0002.2001-05-25.SA_and_HP.ham copy 4.txt'] = -1

"""



def read_files(train_path):
    count = 0
    walked = os.walk(train_path)
    walked = sorted(walked)
    for root, dirs, files in walked:
        files = sorted(files)
        if len(files)>0:
            if root.split('/')[-1] == 'spam':
                for file_name in files:
                    name = root+'/'+file_name
                    O_dict[name] = 1
                    #whole_dir[name] = 1
            else:
                #print(root, len(files), 'ham')
                for file_name in files:
                    name = root+'/'+file_name
                    O_dict[name] = -1
                    #whole_dir[name] = -1

    return O_dict #,whole_dir


def update_weights(set_words, value):
    global bias
    global avg_bias
    global weights
    global avg_weights

    for key in set_words:
        #print(key, value, set_words)
        weights[key]+= value*set_words[key]
        avg_weights[key] += value*count*set_words[key]
        bias +=value
        avg_bias += value*count


def train_model():
    global count
    global avg_weights
    global weights
    global bias
    global avg_bias

    
    for _ in range(max_iter):
        for file, value in O_dict.items():
            if file.split('/')[-1] != '.DS_Store':
                #print(file)
                document = open(file, "r", encoding="latin1")
                set_words = dict()
                alpha = bias
                for line in document:
                    line = line.split(' ')

                    # preprocess
                    for word in line:
                        ###word = word.lower()

                        # punc is not removed - >, --, *, (, )
                        # handling numbers
                        ###if word.isnumeric():
                            #word = "%NUMBER%"

                        if word not in weights:
                            weights[word] = 0
                            avg_weights[word] = 0

                        # changed
                        alpha += weights[word]

                        if word not in set_words:
                            set_words[word] = 1
                        else:
                            set_words[word]+=1
                ##
                #print(file, value, alpha, len(set_words), bias)

                ## check if change is required <=0
                y_alpha = alpha*value
                if y_alpha <=0:
                    update_weights(set_words, value)
                
                count+=1
                #print(weights, avg_weights, bias, avg_bias, count)
                
    for each_word_weight in avg_weights:
        avg_weights[each_word_weight] = weights[each_word_weight] - ((1/count)*avg_weights[each_word_weight])
        avg_bias = bias - ((1/count)*avg_bias)

    print(len(avg_weights), len(weights))


def write_model(model_filename):
    model_file = open(model_filename, "w", encoding="latin1")

    model_dictionary = dict()
    model_dictionary['weights'] = weights
    model_dictionary['avg_weights'] = avg_weights
    model_dictionary['bias'] = bias
    model_dictionary['avg_bias'] = avg_bias

    model_file.write(str(model_dictionary))
    model_file.close()


if __name__ == "__main__":
    #train_path = './train/'
    train_path = sys.argv[1]
    model_filename = 'percmodel.txt'
    read_files(train_path)
    train_model()
    write_model(model_filename)
