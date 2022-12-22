from tensorflow import keras
from keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from keras.layers import Bidirectional , Attention, Layer
from keras.activations import *
import tensorflow as tf
from Attention_S2S import attention
from Decoder import Decoder
from Encoder import Encoder

class Model(Layer):
    def __init__(self, Encoder, Decoder):

        super(Model, self).__init__()

        self.encoder = Encoder
        self.decoder = Decoder
        self.trad = []
        
    def call (self, seq, solut) :  #Take as input the source sentence,and its translation 

        sol_len = solut.get_shape()[-1]  
        enc_outp, hidden = self.encoder(inputs=seq)
        self.trad = []

        for i in range(sol_len): # we go through the target sequence because we are trying to to predict the same    
     
            outp, hidden = self.decoder(inputs=[tf.convert_to_tensor(solut[i]), enc_outp, hidden])
            a = tf.math.argmax(outp)
            a = tf.math.argmax(a)
            a = tf.cast(a, dtype = tf.int32)

            # if a>=sol_len : 
            #     a = tf.cast(sol_len-1, dtype = tf.int32)
            self.trad.append(solut[a])

        return tf.convert_to_tensor(self.trad, dtype = tf.int32) 
    

