import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
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

word_dictionary = dict()
tag_dictionary = dict()
batch_size_model = 100
max_length = 100
dictionaries_path = 'pos_dictionaries.pickle'
output_file_path = 'pos_model.pt'


def update_index(wrd_tag, diction):
    if wrd_tag not in diction:
        diction[wrd_tag] = len(diction)+1


def create_indices(sorted_tokens, all_tokens):
    for token in sorted_tokens:
        split = token[0].split('/')
        update_index(split[0],word_dictionary)

    for token in all_tokens:
        split = token[0].split('/')
        update_index(split[1].strip('\n'),tag_dictionary)


def update_freq(token, tokens_freq):
    if token not in tokens_freq:
        tokens_freq[token] = 1
    else:
        tokens_freq[token] += 1


##
def cus_accuracy_score(a,b):
    count=0
    for i in range(len(a)):
        if (a[i]==b[i]).all():
            count+=1
    return count/len(a)

class transform(Dataset):
    def __init__(self, x, y, leng):
        self.x = torch.tensor(x, dtype = torch.long)
        self.y = torch.tensor(y, dtype = torch.long)
        self.length = leng

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.length[idx]

    def __len__(self):
        return len(self.x)

def read_file(path, boolean):
    data = []
    tokens_freq = dict()
    corpus = open(path, encoding='utf-8').readlines()
    write_f = open('validation.txt', "w")
    for line in corpus:
        words = []
        tags = []
        for token in line.split():
            only_tokens = token.lower() #
            splitted = only_tokens.split('/')
            update_freq(only_tokens,tokens_freq)
            words.append(splitted[0])
            tags.append(splitted[1])
        data.append([words, tags])
        write_line = ' '.join(words)
        write_f.write(write_line+'\n')
    write_f.close()

    # Ask
    # top 1000 words, corresponding tags for top 1000 is 30
    if boolean !=2:
        sorted_ls = sorted([(key, item) for key,item in tokens_freq.items()], key= lambda x: x[1], reverse = True)
        sorted_ls_sliced = sorted_ls[:1200]
        create_indices( sorted_ls_sliced, sorted_ls)
    #print(word_dictionary, tag_dictionary)

    #indexes_data = []
    lengths = []
    indexes_words = []
    indexes_tags = []
    for wrds, tgs in data:
        wrds = [word_dictionary[i] if i in word_dictionary else len(word_dictionary)+1 for i in wrds ]
        tgs = [tag_dictionary[i] if i in tag_dictionary else len(tag_dictionary)+1 for i in tgs ]
        lengths.append(len(wrds))

        # manual padding

        if len(wrds)<max_length:
            wrds = wrds+ [0]*(max_length-len(wrds))
            tgs = tgs + [0]*(max_length-len(tgs))
        else:
            wrds = wrds[:max_length]
            tgs = tgs[:max_length]


        #indexes_data.append((wrds, tgs))
        indexes_words.append(wrds)
        indexes_tags.append(tgs)

    transformed_data_train = transform(indexes_words, indexes_tags, lengths)
    data_loader = DataLoader(transformed_data_train,shuffle=False,batch_size=batch_size_model)
    return data_loader


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

def train_model(data_train, data_dev):
    # model Configuration
    embedding_layer_dim = 100
    hidden_layer_nodes = 150
    layer = 1
    model = BiLSTM_tag(embedding_layer_dim,hidden_layer_nodes, layer, len(word_dictionary)+2,  len(tag_dictionary)+2)
    loss_function = nn.CrossEntropyLoss(size_average=True, ignore_index = 0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20

    prev_f1 = -100
    print('hidden layer size: ',hidden_layer_nodes)
    for epoch in range(epochs):
        print('epoch:', epoch)
        epoch_train_pred = []
        epoch_dev_pred = []
        ground_train_tag = []
        ground_dev_tag = []

        for input_words_batch, tags_tensor_batch, length_vector in data_train:
            #print(input_words_batch, tags_tensor_batch, length_vector)
            model.zero_grad()

            out_probs = model(input_words_batch, length_vector)
            #print('here', out_probs.shape, tags_tensor_batch.shape)
            out_probs_flat = out_probs.view(-1,len(tag_dictionary)+2)
            tags_tensor_batch_flat = tags_tensor_batch.view(-1)

            #print(out_probs.shape, tags_tensor_batch.shape)
            loss = loss_function(out_probs_flat,tags_tensor_batch_flat)

            loss.backward()
            optimizer.step()

            pred = torch.argmax(out_probs, dim=2)
            tags_tensor_batch = tags_tensor_batch.view(-1).numpy()
            pred = pred.view(-1).numpy()

            for pred_index in range(len(pred)):
                if tags_tensor_batch[pred_index] != 0:
                    epoch_train_pred.append(pred[pred_index])
                    ground_train_tag.append(tags_tensor_batch[pred_index])

        print(loss)
        print('F1 train', f1_score(ground_train_tag, epoch_train_pred, average='macro'))
        #print('acc', cus_accuracy_score(ground_train_tag, epoch_train_pred))

        acc = 0
        with torch.no_grad():
            for words_dev_batch, tags_dev_batch, len_vector in data_dev:
                #input_words = torch.tensor(words_dev_batch, dtype = torch.long)
                #tags_tensor = torch.tensor(tags_dev_batch, dtype = torch.long)
                out_probs = model(words_dev_batch, len_vector)
                pred = torch.argmax(out_probs, dim=2).view(-1).numpy()
                tags_dev_batch = tags_dev_batch.view(-1).numpy()
                
                for pred_index in range(len(pred)):
                    if tags_dev_batch[pred_index] != 0:
                        epoch_dev_pred.append(pred[pred_index])
                        ground_dev_tag.append(tags_dev_batch[pred_index])
                        if pred[pred_index] == tags_dev_batch[pred_index]:
                            acc+=1
        print(acc, len(ground_dev_tag))
        f1_present = f1_score(ground_dev_tag, epoch_dev_pred, average='macro')
        print('F1 dev', f1_present)
        if f1_present>prev_f1 and epoch >18:
            save_model(model.state_dict(), output_file_path, dictionaries_path)
            prev_f1 = f1_present
        ##############print('acc', cus_accuracy_score(ground_dev_tag, epoch_dev_pred))

    return model

def save_model(model, filename, dictionaries_path):
    save_list = [word_dictionary, tag_dictionary]
    torch.save(model,filename)
    #model2 = torch.load(filename)
    #print(model2)
    with open(dictionaries_path, 'wb') as file:
        pickle.dump(save_list, file)


if __name__ == "__main__":
    if len(sys.argv)>=3:
        train_data_path = sys.argv[1]
        dev_data_path = sys.argv[2]

        dataloader_train = read_file(train_data_path,1)
        dataloader_dev = read_file(dev_data_path,2)

        model = train_model(dataloader_train,dataloader_dev)
    else:
        print("Not Enough Arguments")
