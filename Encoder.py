import tensorflow as tf
from tensorflow import keras
from keras.activations import *
from keras.layers import Dense , Input , Embedding, Dropout, GRU
from keras.layers import Bidirectional , Layer

class Encoder (Layer):
     
     def __init__(self, embdim, intdim, enc_dimh, dec_dim_h, t_drop):
         super(Encoder, self)._init_()
         #ri :output of the reset gate
         self.emb = Embedding(intdim, embdim) #size of the sequence, and dimension of the embedding
         self.RNN =Bidirectional(GRU (enc_dimh, return_sequences=False, return_state = True ))  #Choose the GRU because he returns the hidden states and the outputs
         self.drop = Dropout(t_drop) 
         self.lin = Dense(dec_dim_h, activation = None ) # Because Multilayer perceptron
             
    def call (self, seq) :
         emb_mod  = self.emb(seq) #Embedding step we give to our data different dimension 
            
         emb_mod = self.drop(emb_mod) #Dropout, we want to avoid the overfitting 

         emb_mod = tf.expand_dims(emb_mod, axis=0)
         outp, hidden, hd= self.RNN(emb_mod)   #hid_state, hd    
         # We apply the activation function the last forward and the last backward state 
         # Each of backward and forward state have their own weights matrices
         hidden = tanh(self.lin(tf.concat((hidden, hd),-1)))# tf.concat convert to one dimension, self.lin apply a linear transformation to the data                  
         
         return [outp, hidden]  # We return the outputs and the hidden states 

