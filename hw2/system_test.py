import csv
import re

conv_complete = 0
punctuations = set('''!()[]{};:'"\,<>./?@#$%^&*_~''')

NLU_states = {'REQUEST_FOOD_TYPE': set(['any', 'italian', 'japanese', 'chinese', 'mexican', 'greek']), 'REQUEST_PRICE' : set(['any', 'cheap', 'medium-priced', 'expensive']), 'REQUEST_LOCATION': {'any': [1], 'marina': [ 'del','rey'], 'venice':[1], 'santa': ['monica'], 'korea': ['town'], 'playa': ['vista'], 'hollywood':[1]}, 'EXPLICIT_CONFIRM_LOCATION': set(['yes', 'no']),'EXPLICIT_CONFIRM_FOOD_TYPE':set(['yes', 'no']), 'EXPLICIT_CONFIRM_PRICE': set(['yes', 'no'])  }
states = {'FOOD_TYPE': 'empty', 'PRICE':'empty', 'LOCATION': 'empty'}
RL_states = {'FOOD_TYPE_FILLED': 'no', 'PRICE_FILLED': 'no', 'LOCATION_FILLED': 'no', 'FOOD_TYPE_CONF': 'no', 'PRICE_CONF': 'no', 'LOCATION_CONF': 'no'}
map_action_states= {'REQUEST_FOOD_TYPE': 'FOOD_TYPE', 'REQUEST_PRICE': 'PRICE','REQUEST_LOCATION' : 'LOCATION'}
map_action_RL_states = {'REQUEST_FOOD_TYPE': 'FOOD_TYPE_FILLED', 'REQUEST_PRICE': 'PRICE_FILLED','REQUEST_LOCATION': 'LOCATION_FILLED', 'EXPLICIT_CONFIRM_LOCATION': 'LOCATION_CONF', 'EXPLICIT_CONFIRM_PRICE' : 'PRICE_CONF', 'EXPLICIT_CONFIRM_FOOD_TYPE' : 'FOOD_TYPE_CONF'}

# read model
## check this function after building the model
def read_model(modelname):
    model_dictionary = dict()
    with open(modelname, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for cnt, row in enumerate(reader):
            if cnt != 0:
                key = [int(i) for i in row[:-1]]
                key = tuple(key)
                model_dictionary[key] = row[-1]
    return model_dictionary


# read database
def read_database(filename):
    data_dictionary = dict()
    file_handle = open(filename, 'r')
    data = file_handle.readlines()
    data_matrix = []
    for line_index in range(1,len(data)):
        line = data[line_index]
        split_line = line[:-1].split('\t')
        for i in range(len(split_line)):
            if i!=0 and i!=1:
                split_line[i] = split_line[i].lower()

        combinations_tuple = [tuple([split_line[2], split_line[3], split_line[4]]),tuple([split_line[2], split_line[3], 'any']), tuple([split_line[2], 'any', split_line[4]]),tuple(['any', split_line[3], split_line[4]]), tuple(['any', 'any', 'any']), tuple(['any', 'any', split_line[4]]), tuple(['any', split_line[3], 'any']), tuple([split_line[2], 'any', 'any'])]
        for tuple_comb in combinations_tuple:
            if tuple_comb not in data_dictionary:
                data_dictionary[tuple_comb] = [(split_line[0], split_line[1])]
            else:
                data_dictionary[tuple_comb].append((split_line[0], split_line[1]))

    return data_dictionary

    """
        if tuple([split_line[2], split_line[3], split_line[4]]) not in data_dictionary:
            data_dictionary[tuple([split_line[2], split_line[3], split_line[4]])] = [(split_line[0], split_line[1])]
            data_dictionary[tuple([split_line[2], split_line[3], 'any'])] = [(split_line[0], split_line[1])]
            data_dictionary[tuple([split_line[2], 'any', split_line[4]])] = [(split_line[0], split_line[1])]
            data_dictionary[tuple(['any', split_line[3], split_line[4]])] = [(split_line[0], split_line[1])]
            data_dictionary[tuple(['any', 'any', 'any'])] = [(split_line[0], split_line[1])]
            #print(data_dictionary.keys())

            if (split_line[0], split_line[1]) == ('Canyon Road', '310-235-4636'):
                print(data_dictionary.keys(), 'here')
            data_dictionary[tuple(['any', 'any', split_line[4]])] = [(split_line[0], split_line[1])]
            print(data_dictionary[tuple(['any', 'any', split_line[4]])], 'ini', ('any', 'any', split_line[4]))
            data_dictionary[tuple(['any', split_line[3], 'any'])] = [(split_line[0], split_line[1])]
            data_dictionary[tuple([split_line[2], 'any', 'any'])] = [(split_line[0], split_line[1])]
        else:
            data_dictionary[tuple([split_line[2], split_line[3], split_line[4]])].append((split_line[0], split_line[1]))
            data_dictionary[tuple([split_line[2], split_line[3], 'any'])].append((split_line[0], split_line[1]))
            data_dictionary[tuple([split_line[2], 'any', split_line[4]])].append((split_line[0], split_line[1]))
            data_dictionary[tuple(['any', split_line[3], split_line[4]])].append((split_line[0], split_line[1]))
            data_dictionary[tuple(['any', 'any', 'any'])].append((split_line[0], split_line[1]))
            data_dictionary[tuple(['any', 'any', split_line[4]])].append((split_line[0], split_line[1]))
            print('this',data_dictionary[tuple(['any', 'any', split_line[4]])], ('any', 'any', split_line[4]))
            data_dictionary[tuple(['any', split_line[3], 'any'])].append((split_line[0], split_line[1]))
            data_dictionary[tuple([split_line[2], 'any', 'any'])].append((split_line[0], split_line[1]))

    """


def update_states(key_word, action):
    global states
    global RL_states

    if action in map_action_states:
        states[map_action_states[action]] = key_word
        RL_states[map_action_RL_states[action]] = 'yes'
    else:
        #print('action', 'remove print')
        if key_word == 'yes':
            RL_states[map_action_RL_states[action]] = key_word
        else:
            # look out for this bug

            if action ==  'EXPLICIT_CONFIRM_LOCATION':
                RL_states['LOCATION_CONF'] = 'no'
                RL_states['LOCATION_FILLED'] = 'no'
                states['LOCATION'] = 'empty'

            elif action == 'EXPLICIT_CONFIRM_PRICE':
                RL_states['PRICE_CONF'] = 'no'
                RL_states['PRICE_FILLED'] = 'no'
                states['PRICE'] = 'empty'

            elif action == 'EXPLICIT_CONFIRM_FOOD_TYPE':
                RL_states['FOOD_TYPE_CONF'] = 'no'
                RL_states['FOOD_TYPE_FILLED'] = 'no'
                states['FOOD_TYPE'] = 'empty'
    #print(states, RL_states)

def NLU(text, action):

    text = re.sub(r'\d+', '', text)
    text2 = ""

    for char in text:
        if char not in punctuations:
            text2+=char

    text = text2.split()
    for ind in range(len(text)):
        text[ind] = text[ind].lower()

    keys_dialogue = NLU_states[action]
    new_flag = 0

    for cnt,word in enumerate(text):
        if word in keys_dialogue:
            if action == 'REQUEST_LOCATION' and len(keys_dialogue[word])>=1 and keys_dialogue[word][0]!=1:
                flag = 0
                pass_update = word
                if len(text)<len(keys_dialogue[word])+1:
                    print(len(text), len(keys_dialogue[word])+1)
                    continue
                for index in range(cnt+1,cnt+1+len(keys_dialogue[word])):
                    if keys_dialogue[word][index-cnt-1] != text[index]:
                        flag =1
                    else:
                        pass_update+=(' '+text[index])

                if flag == 0:
                    update_states(pass_update, action)
                    new_flag = 1
            else:
                update_states(word, action) # yes/no confirm
                new_flag = 1

     ## no states change, give previous action hardcode
    return new_flag


def RL_agent(model,acc, output):
    # category lets me check type of text to check for, type of cuisine etc.

    state_seq = ['FOOD_TYPE_FILLED', 'PRICE_FILLED', 'LOCATION_FILLED', 'FOOD_TYPE_CONF', 'PRICE_CONF','LOCATION_CONF']
    key = []
    for i in state_seq:
        if RL_states[i] == 'yes':
            key.append(1)
        else:
            key.append(0)

    key = tuple(key)
    if sum(key) == 6:
        action = 'DATABASE'
    elif output == 0:
        # check here focus LASLDASJDASDL
        action = model[key]
        #print(model[key], action)
    else:
        action = model[key]
    return action

def nlp_generate(action, database):
    action_text_dictionary = {'REQUEST_FOOD_TYPE': 'What kind of food/cuisine do you like to have?', 'REQUEST_PRICE': 'What is the price range you are looking for?' , 'REQUEST_LOCATION': 'Where would you want the restaurant to be located at?', 'EXPLICIT_CONFIRM_LOCATION': 'Okay, you said you would like to go to a restaurant located at '+states['LOCATION']+' ,right?', 'EXPLICIT_CONFIRM_PRICE': 'Okay, you said you would like to go to a restaurant of price range '+states['PRICE']+' , right?', 'EXPLICIT_CONFIRM_FOOD_TYPE': 'Okay, you said you were looking for a food-type/cusine of '+states['FOOD_TYPE']+' , right?'}

    global conv_complete

    if action == 'DATABASE':
        key_generate = (states['FOOD_TYPE'], states['PRICE'], states['LOCATION'])
        if key_generate in database:
            database_text = database[key_generate]
            out_text = 'I found '+str(len(database_text))+' restaurant/s matching your query. '
            for value in database_text:
                out_text+= value[0]+' is a/an '+states['PRICE']+' price range restaurant with food-type '+ states['FOOD_TYPE']+' ,it is in '+states['LOCATION']+' . '+'The phone number is '+value[1]+'. '
        else:
            out_text = 'I found '+'0'+' restaurant/s matching your query.'
        conv_complete = 1
    else:
        out_text = action_text_dictionary[action]
    return out_text

def main():

    model_name = 'policy-submitted.csv'
    model = read_model(model_name)
    database_name = 'restaurantDatabase.txt'
    database = read_database(database_name)
    action = 'None'
    output = -1

    while conv_complete == 0:
        action = RL_agent(model,action, output)
        # set token as 1 when all states are full
        nlp_gen_text = nlp_generate(action, database)
        print('System: ',nlp_gen_text)
        if conv_complete != 1:
            text_input = input('User: ')
            output = NLU(text_input, action)
        else:
            break


if __name__== "__main__":
  main()
