import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, Dropout, GRU
from keras.layers import Layer
from keras.activations import *
import numpy as np 
from Attention import Attention_S2S

class Decoder (Layer) :

    def __init__(self,  size_out, size_emb, enc_dimh, t_drop, Attention_trad,max_hid_lay= 500): 
        
        super(Decoder, self).__init__()
        
        self.attention = attention
        self.embedding = Embedding(size_out, size_emb) #Embedding matrix of the target word
        self.rnn = GRU (enc_dimh)
        self.dropout = Dropout(t_drop)
        
        self.size_out = size_out
        self.max_hid_lay = max_hid_lay
        self.enc_dimh = enc_dimh 
        self.size_emb = size_emb
        
    def build(self):
        w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001**2)
        #Weight matrices  
        self.W0 = tf.Variable(initial_value = w_init(shape=(self.size_out,self.max_hid_lay,self.units),), trainable=True)

        self.U0 = tf.Variable(initial_value = w_init(shape=(2*self.max_hid_lay,self.enc_dimh,self.units),), trainable=True)
        self.V0 = tf.Variable(initial_value = w_init(shape=(2*self.max_hid_lay,self.size_emb,self.units),), trainable=True)
        self.C0 = tf.Variable(initial_value = w_init(shape=(2*self.max_hid_lay,2*self.enc_dimh,self.units),), trainable=True)
 
    def call (self, entr, hidden, outp_enc, iterat ) :
        emb =  self.embedding(entr)
        emb = Dropout(emb)
        attr = self.attention.call(hidden, outp_enc ) #We compute the attention between this two terms 
        outp, hidden = self.rnn(emb)
        t = self.U0 * hidden + self.V0 * emb * outp + self.C0 * attr
        ti = np.max(t[2*iterat-1:2*iterat])
        prediction = np.exp(outp.T*self.W0*ti)  
        
        return prediction, hidden #We send the predicted word yi, and new values of the hidden states 
    
