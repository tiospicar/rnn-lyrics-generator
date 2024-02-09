# rnn-lyrics-generator

Recurrent neural network that can generate song lyrics. Uses Long Short-Term Memory (LSTM) layers.
Trained on spotify's song lyrics data with 57494 different songs.

# Requirements

Module  | Version
------------- | -------------
keras  | 2.9.0
tensorflow  | 2.11.0
(CUDA versions only) tensorflow-gpu  | 2.9.0
numpy  | 1.21.6
pandas  | 1.3.5

# Instructions

Clone this repository. Run *main.py* and network should start learning. After training is done, model should be saved as *model1.h5*. Current configuration will give you poor results, you can play with bellow variables to find optimal coniguration or you can use my pretrained models *model2_61_val_accu_1m_dataset.h5* or *model1_51_val_accu_500k_dataset.h5*.

Variables you can change:
* TRAIN_DATA_LEN - Number of samples for training
* VALIDATE_DATA_LEN - Number of samples for validating
* BATCH_SIZE
* EPOCHS
* TEMPERATURE - Controls randomness. More temperature gives more randomness when sampling next character. Lower temperature gives more predictable but repetitive results. Optimal between 0.4 and 0.6. Not used in training, only when generating.

Variables I used that produced good results (this configuration takes around 8 hours of training on two T4 GPUs with 61% accuracy on the validation set):
* TRAIN_DATA_LEN = 1000000
* VALIDATE_DATA_LEN = 20000
* BATCH_SIZE = 256
* EPOCHS = 100

You can also change number and size of LSTM layers. This architecture uses 3 layers each with 256 units.
If your GPU can use CUDA then you can use CuDNNLSTM layers instead of LSTM layers for faster training (requires *pip tensorflow-gpu* to be installed).

# Examples

With the temperature 0.5:

INPUT

    "new money, suit and tie
    i can read you like a magazine
    ain't it funny? rumors fly
    and i know you hea"

GENERATED

    "new money, suit and tie
    i can read you like a magazine
    ain't it funny? rumors fly
    and i know you hear uhe tale his heart in a little boy  
    and a gane time will see her to the world  
    and the crazy things are see the drink  
    and he takes a star and lose that he hear  
    he was all the ton  
      
    i don't know what a waiting on the bitd  
    and the day we do, it's gone  
    so something well i got to be  
    i'm a craze and then i got to go  
      
    i want to see you to me  
    i'm a body of down  
    for oret you and my heart  
    i don't know what i need  
    it makes me tell you it is a song  
    i want you to see it  
    to say it's a lov  
    that we don't want to do  
      
    we can be come on  
    and i don't know what i can  
    i can't tell you  
    i can tell you  
    i want to know  
    but i want to hold on  
    i see you  
    with you  
      
    i want to know  
    and when i am  
    i have so long  
    i want to be a stor"

With the temperature 0.1:

INPUT:

    "new money, suit and tie
    i can read you like a magazine
    ain't it funny? rumors fly
    and i know you hea"

GENERATED

    "new money, suit and tie
    i can read you like a magazine
    ain't it funny? rumors fly
    and i know you hear the stars of the stars and i was a little bit of the stars  
      
    i want to be the one that i can see  
    i want to be the one that i can  
    i want to be the one that i can see  
      
    i want to be the sea  
    i want to be the one  
    i want to be the one  
    i want to be the one  
      
    i want to be the beattiful song  
    i want to see you the way  
    i want to be the one that i can  
    i want to be the one that i want  
    i want to be a little bit of the street  
      
    i want to be the one that i can see  
    i want to be the one that i want to be  
    i want to be the one that i want to be  
      
    i want to be the one that i can see  
    i want to be the one that i want to be  
    i want to be the one that i don't know  
    i want to be the sound of the stn  
      
    i want to be the one that i can see  
    i w"

With the temperature 0.9:

INPUT:

    "new money, suit and tie
    i can read you like a magazine
    ain't it funny? rumors fly
    and i know you hea"

GENERATED:

    "new money, suit and tie
    i can read you like a magazine
    ain't it funny? rumors fly
    and i know you hear you...  
      
    be the farth of a different wise  
    you're loighng your walles and let more face  
    kust a rtade of all these all, sure heart down  
    you cenoo her alone rtase and kook now  
    would you wrrn the head with a doon  
    i can't hear the orhooe of a blood  
      
    i eally with my ereams  
    had a long dndlnr  
    she was twila shame  
    and my heart seems to think  
    come on
    
    the gloom to hold le  
    i really love i'm oh rtill  
    just puer on the bad natuin tsn  
    uhat's give it and rould  
    there is gonna love you  
    i'm fine you, i can say  
    but you will blight you that  
    oh, gond longno' i fornd never be night on the dark  
    tell me oh and wou forget me' craadles you happened to you  
      
    oh (nh!  
    we dar blood and i arkan on the eieth mattir  
    your feeling is like longly walk"
