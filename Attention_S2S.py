from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import *
import tensorflow as tf
import numpy as np 

class attention_trad (Layer) :
    def __init__(self, enc_outdim, dec_hdim) :    #We need the hidden satate of the decoder and the encoder 
        #We apply a linear transformation 
        super(attention_trad, self).__init__()
        
    def build (self, inp_shape = dec_hdim) : #State of the layer (weights)
        w_init = tf.keras.initializers.Zeros() 
        #Weight matrices 
        self.Va = tf.Variable (initial_value=w_init(shape=(inp_shape[-1], self.units),) trainable=True)#shape : n, with n : number of hidden states 
        self.Wa = tf.Variable (initial_value = w_init(shape=(inp_shape[-1]*inp_shape[-1],self.units),) trainable=True) #shape : n*n
        self.Ua = tf.Variable (initial_value = w_init(shape=(inp_shape[-1]*2inp_shape[-1],self.units),)trainable=True) #shape : n*2n
    def call (self, outp_enc, hidden_dec ) : #hidden : hj, hidden state of the j th word (from the encoder), out_dec = hidden state decoder 
        #We compute the attention : 
        a = self.Va.T*tanh(self.Wa*hidden_dec[-1]+self.Ua*outp_enc)  #hidden_dec[-1] : decoder hidden state of i-1, (one value)
        alpha = tf.nn.softmax(a) #We normalize the output probabilities
         
        c = np.sum(alpha*hidden_dec) #context 
        return c
 
