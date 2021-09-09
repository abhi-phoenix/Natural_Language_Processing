import sys
import os
import math
import ast
import random

test_dir = list()

def read_files(train_path):    
    count = 0 
    for root, dirs, files in os.walk(train_path):
        if len(files)>0:
            ## next line is required or not? test in vocareum
            #files.remove('.DS_Store')
            for file_name in files:
                if file_name != '.DS_Store':
                    test_dir.append(root+'/'+file_name)


def read_model(model_filename):
    model_file = open(model_filename, "r", encoding="latin1")
    lines = model_file.readlines()
    for line in lines:
        model_data = ast.literal_eval(line)
    return model_data
        
                    
def predict(model, output_file):

    output_file_pointer = open(output_file, "w", encoding="latin1")
    bias = model['bias']
    avg_bias = model['avg_bias']
    weights = model['weights']
    avg_weights = model['avg_weights']

    cnt = 0 
    for file in test_dir:
        cnt+=1
        document = open(file, "r", encoding="latin1")
        alpha = avg_bias
        
        for line in document:
            line = line.split()
            for word in line:
                #word = word.lower()

                # punc is not removed - >, --, *, (, )
                # handling numbers
                #if word.isnumeric():
                    #word = "%NUMBER%"

                if word not in weights:
                    continue
                else:
                    alpha += avg_weights[word]
                    #print(avg_weights[word])

        if alpha > 0:
            label = "spam"
        else:
            label = "ham"
                                        
        # check this line
        if cnt == len(test_dir):
            output_file_pointer.write('%s\t%s' % (label,file))
        else:
            output_file_pointer.write('%s\t%s\n' % (label,file))
        
                                  
if __name__ == "__main__":
    # / inlcuded or not? - check in vocareum about paths returned
    #test_path = 'dev'
    test_path = sys.argv[1]
    output_file = 'percoutput.txt'
    model_filename = 'percmodel.txt'
    
    read_files(test_path)
    model = read_model(model_filename)
    #print(model)
    predict(model, output_file)
    
    
    

    


