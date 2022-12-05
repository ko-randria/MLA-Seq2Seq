from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import *
import tensorflow as tf
import numpy as np 

class Attention_trad (Layer) :
    def __init__(self, enc_outdim, dec_hdim) :     #We need the hidden state of the decoder and the encoder 

        super(Attention_trad, self).__init__()
        
        # initialize the weights matrices (state of the layer)
        inp_shape = dec_hdim
        # Matrices Ua and Wa -> Gaussian distribution of mean 0 and variance of 0.001**2
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001**2)
        self.Ua = initializer(shape=([inp_shape[-1],inp_shape[-1]],self.units), trainable=True) #shape : n*2n
        self.Wa = initializer(shape=([inp_shape[-1],inp_shape[-1]],self.units), trainable=True) #shape : n*n
        
        # Va is initialized at zero
        v_init = tf.keras.initializers.Zeros() 
        self.Va = v_init(shape=(inp_shape[-1], self.units), trainable=True) #shape : n, with n : number of hidden states 
        
    def call (self, outp_enc, hidden_dec ) : # hidden state of the decoder, outp_enc = outputs of the encoder
        #We compute the attention : 
        a  = self.Va.T * tanh(self.Wa*hidden_dec[-1]+self.Ua*outp_enc )  # hidden_dec[-1] : decoder hidden state of i-1,
        alpha = tf.nn.softmax(a) # We normalize the output probabilities
         
        c = np.sum(alpha*hidden_dec) # context 
        return c
