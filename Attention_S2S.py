import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer
from keras.activations import *
import numpy as np 

class attention (Layer) :
    def __init__(self, dec_hdim) :    #We need the hidden satate of the decoder and the encoder 
        #We apply a linear transformation 
        super(Attention_trad, self).__init__()
        self.inp_shape = dec_hdim
        
    def build (self, inp_shape = dec_hdim) : #State of the layer (weights)
        w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001**2)
        #Weight matrices 
        self.Wa = tf.Variable(initial_value = w_init(shape=(self.inp_shape[-1]*self.inp_shape[-1],self.units),), trainable=True) #shape : n*n
        self.Ua = tf.Variable(initial_value = w_init(shape=(self.inp_shape[-1]*2*self.inp_shape[-1],self.units),),trainable=True) #shape : n*2n
        # Va is initialized at zero
        v_init = tf.keras.initializers.Zeros() 
        self.Va = tf.Variable(initial_value=v_init(shape=(self.inp_shape[-1], self.units),), trainable=True)#shape : n, with n : number of hidden states
        
    def call (self, outp_enc, hidden_dec ) : #hidden : hj, hidden state of the j th word (from the encoder), out_dec = hidden state decoder 
        #We compute the attention : 
        a = self.Va.T * tanh(self.Wa *hidden_dec[-1] + self.Ua * outp_enc)  #hidden_dec[-1] : decoder hidden state of i-1, (one value)
        alpha = tf.nn.softmax(a) #We normalize the output probabilities
         
        c = np.sum(alpha*hidden_dec) #context 
        return c
 
