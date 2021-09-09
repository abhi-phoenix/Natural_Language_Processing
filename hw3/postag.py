import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data.sampler import SequentialSampler
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import sys
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import statistics as stats
import pickle

max_length = 100
embedding_layer_dim = 100
hidden_layer_nodes = 150
layer = 1

class transform(Dataset):
    def __init__(self, x, leng):
        self.x = torch.tensor(x, dtype = torch.long)
        self.length = leng

    def __getitem__(self, idx):
        return self.x[idx], self.length[idx]

    def __len__(self):
        return len(self.x)

class BiLSTM_tag(nn.Module):

    def __init__(self, embedding_dimension, hidden_dimension, layers, input_length, tags_output_dim):
        super(BiLSTM_tag, self).__init__()
        self.input_size = input_length
        self.hidden_dimension = hidden_dimension
        self.word_embeddings = nn.Embedding(self.input_size, embedding_dimension, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dimension, hidden_dimension,layers,batch_first = True, bidirectional =True)
        self.linear_out = nn.Linear(2*hidden_dimension, tags_output_dim)

    def forward(self, tokens_in, length_vector):
        embeddings = self.word_embeddings(tokens_in)
        packed_embedded = pack_padded_sequence(embeddings, length_vector,batch_first=True, enforce_sorted=False)
        #print(embeddings.shape)
        lstm_output, _ = self.lstm(packed_embedded)
        out_padded, out_lengths = pad_packed_sequence(lstm_output, batch_first=True, total_length=100 ) #
        #print(lstm_output.shape)
        tags_out = self.linear_out(out_padded)
        #print(tags_out.shape)
        #tag_prob = F.log_softmax(tags_out, dim=1)
        #print('this',tag_prob.shape)
        return tags_out

def load_model(filename, model_name):
    with open(filename,'rb') as file:
        dictionaries = pickle.load(file)
    word_dictionary, tag_dictionary = dictionaries
    model = BiLSTM_tag(embedding_layer_dim,hidden_layer_nodes, layer, len(word_dictionary)+2,  len(tag_dictionary)+2)
    model.load_state_dict(torch.load(model_name))
    return model, word_dictionary, tag_dictionary


def read_write_file(read_file, write_file, model, word_dictionary, tag_dictionary):
    my_inverted_dict = dict(map(reversed, tag_dictionary.items()))
    write_f = open(write_file, "w")
    corpus = open(read_file, encoding='utf-8').readlines()
    split_lines = []
    data  =[]
    lengths = []
    for line in corpus:
        split_line = line.split()
        split_lines.append(split_line)
        lengths.append(len(split_line))
        id_vector = [word_dictionary[i] if i in word_dictionary else len(word_dictionary)+1 for i in split_line]
        if len(id_vector)<max_length:
            id_vector = id_vector+ [0]*(max_length-len(id_vector))
        else:
            id_vector = id_vector[:max_length]
        data.append(id_vector)

    transformed_data = transform(data, lengths)
    data_loader = DataLoader(transformed_data,shuffle=False)
    #print(data)
    ####
    
    with torch.no_grad():
        for cnt, data_point in enumerate(data_loader):
            input_words_batch, length_vector = data_point
            print(length_vector, input_words_batch)
            out_probs = model(input_words_batch, length_vector)
            pred = torch.narrow(torch.argmax(out_probs, dim=2), 1,0,length_vector[0]).numpy()[0]
            #print(pred, tag_dictionary)
            pred = [split_lines[cnt][i]+'/'+my_inverted_dict[pred[i]] for i in range(len(pred))]
            write_line = ' '.join(pred)
            if cnt != len(data)-1:
                write_f.write(write_line+'\n')
            else:
                write_f.write(write_line)
            #print(write_line)
    write_f.close()

    """ 
    with torch.no_grad():
        for cnt, data_tokens in enumerate(data):
            print(data_tokens, split_lines[cnt])
            input_words = torch.tensor([data_tokens], dtype = torch.long)
            out_probs = model(input_words, [lengths[cnt]])
            pred = torch.argmax(out_probs, dim=2).numpy()
            print(pred.shape)
               
    """

if __name__ == "__main__":
    if len(sys.argv)>=3:
        valid_data_path = sys.argv[1]
        write_file_path = sys.argv[2]
        load_model_file = 'pos_model.pt'
        dictionaries_path = 'pos_dictionaries.pickle'

        model, word_dictionary, tag_dictionary = load_model(dictionaries_path,load_model_file)
        read_write_file(valid_data_path, write_file_path, model, word_dictionary, tag_dictionary )

    else:
        print("Not Enough Arguments")

  

#print(model.state_dict())
