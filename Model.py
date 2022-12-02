from tensorflow.keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from tensorflow.keras.layers import Bidirectional , Attention, Layer
from tensorflow.keras.activations import *
import tensorflow as tf
import numpy as np
import Attention, Decoder, Encoder

class Model(Layer):
    def __init__(self, seq, encoder, decoder, param):

        super(Model, self)._init_()

        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, source):
        # compute the encoder outputs and hidden states
        encoder_outp, hidden = encoder.call(source)

        # Decoder

