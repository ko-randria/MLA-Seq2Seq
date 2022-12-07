from tensorflow.keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from tensorflow.keras.layers import Bidirectional , Attention, Layer
from tensorflow.keras.activations import *
import tensorflow as tf
import numpy as np
from Attention_S2S import Attention_trad
from Decoder import Decoder
from Encoder import Encoder

class Model(Layer):
    def __init__(self, seq, Encoder, Decoder):

        super(Model, self).__init__()

        self.encoder = Encoder
        self.decoder = Decoder
    
    def call (self, seq, solut) :  #Take as input the source sentence,and its translation  
        sol_len = solut.shape[0]  # We extract the lenght of the translation   
        len_seq = seq.shape[0] #  We extract the lenght of the source 
        batch_size = seq.shape[1]   #  The batch size 
        val_size = self.decoder.size_out
        dec_outp = tf.zeros (sol_len, batch_size, val_size)
        enc_outp, hidden = self.encoder(seq) # We get the outputs andn the hidden states of the encoder 
                
        
        for i in range(sol_len):#we go through the target sequence, we try to predict the same   
            outp, hidden = self.decoder(solut, hidden, enc_outp) # we give the decoder the target, the hidden states and outputs of the encoder 
            
            dec_outp[i] = outp # we store  the predictions  of the decoder  
            max_pred = outp.argmax(1) # We select the max prediction

            entr =  max_pred #We replace the target by our predicted word

        return dec_outp 
    

