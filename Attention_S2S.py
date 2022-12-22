import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Dense
from keras.activations import *

class attention (Layer) :
    def __init__(self, dec_hdim) :    #We need the hidden state of the decoder and the encoder 
        #We apply a linear transformation 
        super(attention, self).__init__()
        self.inp_shape = dec_hdim
    
        w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001**2)

        #Weight matrices 
        self.Wa = tf.Variable(initial_value = w_init(shape=(1,self.inp_shape)), trainable=True) 
        self.Ua = tf.Variable(initial_value = w_init(shape=(1,2*self.inp_shape)),trainable=True) 
        # Va is initialized at zero
        v_init = tf.keras.initializers.Zeros() 
        self.Va = tf.Variable(initial_value=v_init(shape=(self.inp_shape)), trainable=True)
        self.dens = Dense(dec_hdim, kernel_initializer = v_init,  activation = None )  
    
    def call (self, outp_enc, hidden ) : # outp_enc : output encoder ; hidden : hj, hidden state of the j th word (from the encoder)
        #We compute the attention :
        hidden = tf.squeeze (hidden)
        outp_enc = tf.squeeze (outp_enc)

        a = self.dens(tanh(tf.concat([self.Wa *hidden, self.Ua * outp_enc],1)))

        alpha = tf.nn.softmax(a, axis = 1)  #We normalize the output probabilities 
        c = alpha*hidden #context 
        c = tf.reduce_sum(c, axis=0)
        c = tf.expand_dims(c,axis=0)
        return c

 
