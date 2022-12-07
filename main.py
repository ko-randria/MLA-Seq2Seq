import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.activations import *
from keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from keras.layers import Bidirectional , Attention, Layer
from Encoder import Encoder 
from Decoder import Decoder
from Attention import Attention_S2S
from Model import Model
import model_training

import tensorflow_text as tf_text
import glob

HIDDEN_LAYER = 1000 # hidden lay
WORD_EMBEDDING =  620 # word embedding
MAXOUT_HIDDEN_LAYER = 500 # maxout hidden layour in the deep outpu
HIDDEN_UNIT = 1000 # The number of hidden units in the alignments model


# Download the file
path_fr = glob.glob('./training/test_fr.txt')
path_en = glob.glob('./training/test_en.txt') 

def load_data(path):
    sentences = []
    with open(path,'r',encoding="utf8") as file:
        for line in file:
            sentences.append([line])

    return np.array(sentences)

source_raw = load_data(path_fr[0])
target_raw = load_data(path_en[0])

# TEXT PREPROCESSING
# Standardization (to strip the accent from the French dataset and keep punctuation) 
def tf_Standardization(text):
  # Split accented characters.
  text = tf_text.normalize_utf8(text, 'NFKD')
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z A-Z.?!,¿\']', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿:;\']', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)
    
  return text

source_raw = [tf_Standardization(x) for x in source_raw]
target_raw  = [tf_Standardization(x) for x in target_raw]

tokenizer = tf_text.WhitespaceTokenizer()
source_raw  = [tokenizer.tokenize(x) for x in source_raw ]
target_raw  = [tokenizer.tokenize(x) for x in target_raw ]

print("\n")
print("TEST START\n")
print(source_raw)
print(target_raw)
print("\n")
print("TEST END\n")

# THERE ARE STILL PROBLEM TO BE SOLVED THERE, IT DOESN'T WORK YET
#INPUT_DIM = 30
#OUTPUT_DIM = 30
#DROPOUT = 0.5

#encoder = Encoder(WORD_EMBEDDING , INPUT_DIM , HIDDEN_UNIT, HIDDEN_UNIT, DROPOUT)
#attention = Attention_S2S(HIDDEN_UNIT)
#decoder =Decoder(OUTPUT_DIM, WORD_EMBEDDING, HIDDEN_UNIT, DROPOUT, attention )

#model = Model(encoder,decoder)

#inputs= tf.keras.layers.Input(shape=(30,))

#S2S_model = tf.keras.Model(inputs=inputs, outputs=model)
#S2S_model.summary()

