import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, Dropout, GRU, Dense
from keras.layers import Layer
from keras.activations import *
from Attention_S2S import attention

class Decoder (Layer) :

    def __init__(self,  size_out, size_emb, enc_dimh, dec_dimh, t_drop, attention, max_hid_lay= 500): 
        
        super(Decoder, self).__init__()
        self.attention = attention 
        self.embedding = Embedding(size_emb, size_emb) # Embedding matrix of the target word

        w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001**2)
        self.rnn = GRU (1000, kernel_initializer = w_init, return_sequences=False, input_shape = (1,1000))
        self.dropout = Dropout(t_drop)
        self.max_hid_lay = max_hid_lay
        self.enc_dimh = enc_dimh 
        self.size_emb = size_emb
        self.dense = Dense (size_out,  activation = None )  # Dense must have the lenght of the target as the size output 

    def call (self, entr, outp_enc, hidden) :
        emb =  self.embedding(entr)

        emb_r = tf.expand_dims(emb,axis=0) # emb_r : [1,len(entr),size_emb]
        emb_r = self.dropout (emb_r) # Dropout step
        attr = self.attention.call(outp_enc, hidden) # We compute the attention between this two terms 

        hid_dec = self.rnn (emb_r )

        predict = self.dense (tf.concat ([hid_dec, tf.expand_dims(tf.squeeze(emb_r), axis=0), attr],1)) 

        return [predict, hid_dec]

    
