from tensorflow.keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from tensorflow.keras.layers import Bidirectional , Attention, Layer
from tensorflow.keras.activations import *
import tensorflow as tf
import numpy as np
from Attention_S2S import Attention_trad
from Decoder import Decoder
from Encoder import Encoder

class Model(Layer):
    def __init__(self, seq, Encoder, Decoder):

        super(Model, self).__init__()

        self.encoder = Encoder
        self.decoder = Decoder
    
    def call(self, source, target):
        # compute the encoder outputs and hidden states
        encoder_outp, hidden = self.encoder.call(source)

        # Decoder
        initializer = tf.keras.initializers.Zeros()
        model_output = initializer(shape=(target.shape[0],))
        for i in range(len(target)):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder.call(model_output, hidden, encoder_output, i )
            
            #place predictions in a tensor holding predictions for each token
            model_output[i] = output
        
        return model_output

