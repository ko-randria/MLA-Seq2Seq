from tensorflow.keras.layers import Embedding, Dropout, GRU
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import *
import tensorflow as tf
import numpy as np 

class Decoder (Layer) :

    def __init__(self,max_hid_lay= 500,  size_out, size_emb, enc_dimh, dec_dimh, t_drop, attention): 
        
        super(Decoder, self).__init__()
        
        self.attention = attention_trad
        self.embedding = Embedding(size_out, size_emb) #Embedding matrix of the target word
        self.rnn = GRU (enc_dimh)
        self.dropout = Dropout(t_drop)
        
        # initialize the weights matrices (state of the layer)
        # Weights matrice other than Ua and Wa -> Gaussian distribution of mean 0 and variance of 0.01**2
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01**2)
        self.W0 = initializer(shape=([size_out,max_hid_lay], self.units), trainable=True)#shape : n, with n : number of hidden states, max_hid_lay = 500, with max_hid_lay : size of the maxout hidden layer in the deep output, symbole l in the article 
        self.U0 = initializer(shape=([2*max_hid_lay,enc_dimh] ,self.units), trainable=True)
        self.V0 = initializer(shape=([2*max_hid_lay,size_emb] ,self.units), trainable=True)
        self.C0 = initializer(shape=([2*max_hid_lay,2*enc_dimh] ,self.units), trainable=True)
 
    def call (self, entr, hidden, outp_enc, iterat ) :
        emb =  self.embedding(entr)
        emb = dropout(emb)
        attr = self.attention(hidden, outp_enc ) #We compute the attention between this two terms 
        outp, hidden = self.rnn(emb)
        t = self.U0 * hidden + self.V0 * emb * outp + self.C0 * attr
        ti = np.max(t[2*iterat-1:2*iterat])
        prediction = np.exp(outp.T*self.W0*ti)  
        
        return prediction, hidden #We send the predicted word yi, and new values of the hidden states 
    
