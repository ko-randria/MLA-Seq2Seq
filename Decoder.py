import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, Dropout, GRU, Dense
from keras.layers import Layer
from keras.activations import *
import numpy as np 
from Attention_S2S import attention

class Decoder (Layer) :

    def __init__(self,  size_out, size_emb, enc_dimh, dec_dimh, t_drop, attention, max_hid_lay= 500): 
        
        super(Decoder, self).__init__()
        self.attention = attention 
        self.embedding = Embedding(size_emb, size_emb) #Embedding matrix of the target word
        w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001**2)
        self.rnn = GRU (dec_dimh, kernel_initializer = w_init ) 
        self.dropout = Dropout(t_drop)
        self.max_hid_lay = max_hid_lay
        self.enc_dimh = enc_dimh 
        self.size_emb = size_emb
        self.dense = Dense (size_out,  activation = None )  #Dense must have the lenght of the target as the size output 

    def call (self, entr, hidden, outp_enc, iterat) :

        emb =  self.embedding(entr)

        emb_r = tf.expand_dims(emb,axis=0) #emb_r : [1,len(entr),size_emb]
        emb_r = self.dropout (emb_r) #Dropout step
        attr = self.attention.call( hidden,outp_enc) #We compute the attention between this two terms 
        
        #The decoder given the 
        emb = tf.expand_dims(emb_r,axis=0)
        hid_dec = self.rnn(emb) 
        
        #We concatenate the three elements : attention,the output RNN decoder, the embedded world
        predict = self.dense (tf.concat ([hid_dec, emb_r, attr],1)) #tf.concat : [len(entr), 2*enc_dimh+size_emb]

        #ti = np.max(t[:, 2*iterat-1:2*iterat])
        #print(ti)
        #prediction = np.exp(tf.transpose(entr)*t)  
        
        #prediction = tf.math.reduce_max(predict)

        return predict, hidden #We send the predicted word yi, and new values of the hidden states 
    
