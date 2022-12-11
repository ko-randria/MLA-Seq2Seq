import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dense
from keras.activations import *
import numpy as np 

class attention (Layer) :
    def __init__(self, dec_hdim) :    #We need the hidden satate of the decoder and the encoder 
        super(attention, self).__init__()
        self.inp_shape = dec_hdim
         #Weight matrices 
        w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001**2)
        self.Wa = tf.Variable(initial_value = w_init(shape=(self.inp_shape*self.inp_shape)), trainable=True) #shape : n*n
        self.Ua = tf.Variable(initial_value = w_init(shape=(self.inp_shape*2*self.inp_shape)),trainable=True) #shape : n*2n
        # Va is initialized at zero
        v_init = tf.keras.initializers.Zeros() 
        self.Va = tf.Variable(initial_value=v_init(shape=(self.inp_shape)), trainable=True)#shape : n, with n : number of hidden states
        self.dens = Dense (dec_hdim, kernel_initializer = v_init,  activation = None ) #Input : [3000] because [2*enc_dimh+enc_dimh], the output size : dec_hdim,  

    def call (self, hidden_dec,outp_enc ) : #hidden : hj, hidden state of the j th word (from the decoder), out_dec = output of the encoder 
        #We compute the attention : 
        #outp_enc : shape :  2000 // hidden_dec : 1000
        a = self.dens(tanh(tf.concat([self.Wa *hidden, self.Ua * outp_enc],1))) # a : [1,dec_hdim], #The dense layer is very important we pass from a dim 3000 to 1000 as an output size         
        alpha = tf.nn.softmax(a, axis = 1) #We normalize the output probabilities
        c = tf.reduce_sum(alpha*hidden_dec, axis = 0) #context 
        c = tf.expand_dims(c,axis=0)
        return c
 
