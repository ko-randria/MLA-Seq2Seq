import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.activations import *
from keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from keras.layers import Bidirectional , Attention, Layer
from Encoder import Encoder 
from Decoder import Decoder
from Attention_S2S import attention
from Model import Model
from model_training import model_training
import tensorflow_text as tf_text
import glob
from tensorflow.python.keras.layers import Lambda
#from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer



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


# Parameters of the Model
HIDDEN_LAYER = 1000 # hidden lay
WORD_EMBEDDING =  620 # word embedding
MAXOUT_HIDDEN_LAYER = 500 # maxout hidden layour in the deep outpu
HIDDEN_UNIT = 1000 # The number of hidden units in the alignments model
HIDDEN_DEC = 1000
param= [HIDDEN_LAYER,WORD_EMBEDDING, MAXOUT_HIDDEN_LAYER,  HIDDEN_UNIT]

# Download the test Dataset to train the model
path_fr = glob.glob('Dataset/test_fr.txt')
path_en = glob.glob('Dataset/test_en.txt')


def load_data(path):
    sentences = []
    with open(path,'r',encoding="utf8") as file:
        for line in file:
            sentences.append([line])

    return np.array(sentences)

source_raw = load_data(path_fr[0])
target_raw = load_data(path_en[0])

# TEXT PREPROCESSING
#English
Token_Target = Tokenizer(num_words=30000)
Token_Target.fit_on_texts(target_raw[:,0])
num_trg = Token_Target.texts_to_sequences(target_raw[:,0])

#We convert the texts to integers vector
#French
Token_Src = Tokenizer(num_words=30000)
Token_Src.fit_on_texts(source_raw[:,0])
num_src = Token_Src.texts_to_sequences(source_raw[:,0])

#Padding : 
num_src  = pad_sequences(num_src, maxlen=30, truncating='post')
num_trg  = pad_sequences(num_trg, maxlen=30, truncating='post')



# THERE ARE STILL PROBLEM TO BE SOLVED IN THIS SECTION : NO GRADIENT WHEN TRAINING 
INPUT_DIM = 30 # number of words
OUTPUT_DIM = 30 
DROPOUT = 0.5
NUM_WORDS = 30000
encoder = Encoder(INPUT_DIM, WORD_EMBEDDING , HIDDEN_UNIT, HIDDEN_UNIT, DROPOUT, NUM_WORDS)
attention = attention(HIDDEN_UNIT)
HIID =tf.zeros((1, 1, 1000))
decoder =Decoder(OUTPUT_DIM, WORD_EMBEDDING, HIDDEN_DEC, HIID,  DROPOUT, attention, NUM_WORDS )

# Input
input1 =  tf.keras.Input(shape=(30),dtype=tf.float32)
input2 = tf.keras.Input(shape=(30),dtype=tf.float32)
entree = tf.keras.Input(shape=(1),dtype=tf.float32) 
input1 = tf.reshape(input1, [30])
input2 = tf.reshape(input2, [30])
entree = tf.reshape(entree, [1])


# BUILDING THE MODEL
# Encoder
encoder_model = tf.keras.models.Model(inputs=[input1], outputs=encoder.call(input1))
print("Encoder model summary")
encoder_model.summary()
out_enc, hidden_enc = encoder_model.output
# Attention
attention_model = tf.keras.models.Model(inputs=[out_enc, hidden_enc], outputs=[attention.call(out_enc, hidden_enc)])
print("Attention model summary")
attention_model.summary()
# Decoder
decoder_model = tf.keras.models.Model(inputs=[entree, out_enc, hidden_enc], outputs=decoder.call(entree, out_enc, hidden_enc))
print("Decoder model summary")
decoder_model.summary()

# Building the Seq2Seq model
model = Model(encoder_model,decoder_model)
S2S_model = tf.keras.models.Model(inputs=[input1,input2], outputs=model.call(input1,input2))
print("Seq2Seq model summary")
S2S_model.summary ()

# Compile the model with the optimizer and loss specified in the article
S2S_model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=10e-6, rho=0.95),
              loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

# Training the model
source = [tf.cast(tf.reshape(num_src[i], [30]),dtype=tf.float32) for i in range(len(num_src))]
target = [tf.cast(tf.reshape(num_trg[i], [30]),dtype=tf.float32) for i in range(len(num_trg))]

#history = S2S_model.fit([source[0], target[0]], target, batch_size=1, epochs=10, validation_split=0.2) 

# TRAINING
train = model_training(param, S2S_model)
train.training_loop(13, source, target, True)
