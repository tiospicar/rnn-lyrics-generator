import numpy as np
import pandas as pd
import random
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dense, Dropout

SEQ_LEN = 100 # Length of RNN sequence input
PRED_LEN = 1 # Length of RNN sequence output
TRAIN_DATA_LEN = 100000 # Number of samples for training
VALIDATE_DATA_LEN = 20000 # Number of samples for validating
BATCH_SIZE = 256
EPOCHS = 100
TEMPERATURE = 0.4 # More temperature gives more randomness when sampling next character. Lower temperature gives more predictable but repetitive results. Optimal between 0.4 and 0.6.
MODEL_OUTPUT_FILE = './models/model.h5'

def preprocess_data(df):
    """ Preprocess data """
    print("Preprocessing data...")

    df = df.dropna() # Remove Nan rows
    df = df.drop(columns=['link', 'song']) # Remove link and song column
    df['text'] = df['text'].str.lower() # Upper case to lower case

    corpus = ''
    for i in range(10000): # We got same result as going through whole dataset
        corpus += df['text'].iloc[i]

    char_set = sorted(set(corpus)) # We created sorted set of unique characters
    char_to_index = dict((c, i) for i, c in enumerate(char_set)) # We create dictionary for charatacter and indexes
    index_to_char = dict((i, c) for i, c in enumerate(char_set)) # We create dictionary for charatacter and indexes

    print("Data preprocessed!")
    return df, char_to_index, index_to_char

def split_data_chars(df, char_to_index):
    """ We split data into datasets for training and testing """

    print("Splitting data...")
    train_x = []
    train_y = []
    validate_x = []
    validate_y = []

    for i in range(TRAIN_DATA_LEN):
        row_index = random.randint(0, len(df) - 1) # We select random song index
        row = df["text"].iloc[row_index]

        seq_index = random.randint(0, len(row) - SEQ_LEN + PRED_LEN + 1)
        x = row[seq_index:seq_index + SEQ_LEN]
        y = row[seq_index + SEQ_LEN:seq_index + SEQ_LEN + PRED_LEN]

        x = [char_to_index[char] for char in list(x)] # We encode the list with char_to_index
        y = [char_to_index[char] for char in list(y)] # We encode the list with char_to_index

        if len(x) == SEQ_LEN and len(y) == PRED_LEN:
            train_x.append(x)
            train_y.append(y)

    for i in range(VALIDATE_DATA_LEN):
        row_index = random.randint(0, len(df) - 1) # We select random song index
        row = df["text"].iloc[row_index]

        seq_index = random.randint(0, len(row) - SEQ_LEN + PRED_LEN + 1)
        x = row[seq_index:seq_index + SEQ_LEN]
        y = row[seq_index + SEQ_LEN:seq_index + SEQ_LEN + PRED_LEN]
        
        x = [char_to_index[char] for char in list(x)] # We encode the list with char_to_index
        y = [char_to_index[char] for char in list(y)] # We encode the list with char_to_index

        if len(x) == SEQ_LEN and len(y) == PRED_LEN:
            validate_x.append(x)
            validate_y.append(y)

    print("Data splitten!")
    return train_x, train_y, validate_x, validate_y
   
def sample_with_temperature(prediction): # We add a little randomness when sampling from learned distribution to avoid repetitive results
    """ Adding randomness when predicting next character """

    prediction = np.log(prediction) / TEMPERATURE # Lower the temperature, predictions will be more confident, higher the temperature we have more randomness
    exp_prediction = np.exp(prediction)
    normalized_prediction = exp_prediction / np.sum(exp_prediction)
    sampled_index = np.random.choice(len(prediction), p=normalized_prediction)
    return sampled_index

def generate_lyrics(model, length=100):
    """ Generate lyrics """

    #input_text = 'all my people from the front to the back nod, back nod\nnow, who thinks their arms are long enough to' # len 100
    #input_text = "new money, suit and tie\ni can read you like a magazine\nain't it funny? rumors fly\nand i know you hea" # len 100
    input_text = "you say my name like i have never heard before\ni'm indecisive but this time i know for sure\ni hope i" # len 100
    
    print("LEN: " + str(len(input_text)))
    generated_text = input_text

    for i in range(length):
        input = [char_to_index[char] / float(len(char_to_index)) for char in list(input_text)]
        
        p = model.predict([input], verbose=0)[0]
        p = np.asarray(p).astype('float64')
        
        index = sample_with_temperature(p)
        next_char = index_to_char[index]
        
        input_text = input_text[1:]
        input_text += next_char
        generated_text += next_char

    print(generated_text)
        
def train():  
    """ Train the model with set parameters """

    train_x, train_y, validate_x, validate_y = split_data_chars(main_df, char_to_index) # Split data into train and validate sets (into chars)
    
    train_x = np.array(train_x)
    train_x = train_x / float(len(char_to_index))
    
    validate_x = np.array(validate_x)
    validate_x = validate_x / float(len(char_to_index))

    train_y = np.array(train_y)
    train_y_one_hot = to_categorical(train_y, num_classes=len(char_to_index))
    
    validate_y = np.array(validate_y)
    validate_y_one_hot = to_categorical(validate_y, num_classes=len(char_to_index))
    
    print("Can use CUDA? ")
    print(tf.test.is_built_with_cuda())
    print("Train X shape ")
    print(train_x.shape)
    print("Train Y one hot shape ")
    print(train_y_one_hot.shape)
    
    model = Sequential()

    model.add(CuDNNLSTM(256, input_shape=(train_x.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.5))
    
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    
    model.add(CuDNNLSTM(256))
    model.add(Dropout(0.5))
    
    model.add(Dense(train_y_one_hot.shape[1], activation='softmax'))
    opt = tf.keras.optimizers.legacy.Adamax(lr=0.001, decay=1e-6)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    
    model.summary()
    
    model.fit(train_x, train_y_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validate_x, validate_y_one_hot))
    model.save(MODEL_OUTPUT_FILE)
    
if __name__ == '__main__':
    main_df = pd.read_csv('./data/spotify_millsongdata.csv') # Read lyrics file
    main_df, char_to_index, index_to_char = preprocess_data(main_df) # Preprocess lyrics

    train() # Comment this line after model is trained
    model = tf.keras.models.load_model(MODEL_OUTPUT_FILE) # Loads model that was trained
    #model = tf.keras.models.load_model('./models/model2_61_val_accu_1m_dataset.h5') # Loads pretrained model
    generate_lyrics(model=model, length=200)