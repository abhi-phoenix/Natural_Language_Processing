import sys
import os
import math

labels = []
cor_class_spam = 0
bel_class_spam = 0
classify_class_spam = 0 

cor_class_ham = 0
bel_class_ham = 0
classify_class_ham = 0 

def read_files(output_file):
    global cor_class_spam 
    global bel_class_spam 
    global classify_class_spam 

    global cor_class_ham 
    global bel_class_ham 
    global classify_class_ham 

    
    file_data = open(output_file, "r", encoding="latin1")
    lines = file_data.readlines()

    for line in lines:
        split_line = line.split('\t')
        if len(split_line) == 2:    
            predicted_label = split_line[0]
            # -2,can directories vary in name path -3?
            actual_label = split_line[1].split('/')[-2]
            #print(predicted_label, actual_label)
            if predicted_label == actual_label:
                if predicted_label == "spam":
                    cor_class_spam +=1
                    bel_class_spam +=1
                    classify_class_spam +=1
                else:
                    cor_class_ham +=1
                    bel_class_ham +=1
                    classify_class_ham +=1
                    
                    
            elif predicted_label != actual_label:
                if actual_label == "spam":
                    bel_class_spam +=1
                else:
                    bel_class_ham +=1

                if predicted_label == "spam":
                    classify_class_spam +=1
                else:
                    classify_class_ham += 1
        else:
            continue

    # divide by zero handle
    if classify_class_spam == 0:
        classifiy_class_spam = 0.01
    if classify_class_ham == 0:
        classify_class_ham = 0.01
    if bel_class_spam == 0:
        bel_class_spam = 0.1
    if bel_class_ham == 0:
        bel_class_ham = 0.1
    
    precision_spam = cor_class_spam/classify_class_spam
    precision_ham = cor_class_ham/classify_class_ham
    
    recall_spam = cor_class_spam/bel_class_spam
    recall_ham = cor_class_ham/bel_class_ham

    if precision_spam+recall_spam == 0:
        precision_spam, recall_spam = 0.1, 0.1
    if precision_ham+recall_ham == 0:
        precision_ham, recall_ham = 0.1, 0.1
        
    f1_spam = (2*precision_spam*recall_spam)/(precision_spam+recall_spam)
    f1_ham = (2*precision_ham*recall_ham)/(precision_ham+recall_ham)
    
    print("Precision of Spam: ", precision_spam, ", Precision of Ham: ",precision_ham)
    print("Recall of Spam: ", recall_spam,", Recall of Ham: ", recall_ham)
    print("F1 of Spam: ", f1_spam,", F1 of Ham: ", f1_ham)

if __name__ == "__main__":
    #output_file = 'nboutput.txt'
    output_file = sys.argv[1]
    read_files(output_file)

    


