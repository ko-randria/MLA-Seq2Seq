from tensorflow.keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from tensorflow.keras.layers import Bidirectional , Attention, Layer
#from tensorflow.keras.activations import tanh 
from tensorflow.keras.activations import *
class encoder (Layer):
     def _init_(self, embdim, intdim, enc_dimh, dec_dim_h, t_drop):
        
             super(encoder, self)._init_()
             #ri :output of the reset gate

             self.emb = Embedding(intdim, embdim) #size of the sequence, and dimension of the embedding
             self.RNN =Bidirectional(GRU (enc_dimh)) #Choose the GRU because he returns the hidden states and the outputs 
             self.drop = Dropout(t_drop)
             self.lin = Dense (dec_dim_h, activation = None ) #output size,  because Multilayer perceptron 
             
     def call (self, seq) :
         emb_mod  = self.emb(seq) #Embedding step we give to our data different dimension 
         
         outp, hid_state = self.RNN(emb_mod)
         
         #We apply the activation function the last forward and the last backward state 
         hidden = tanh(self.lin(tf.concat((hid_state[-2,:,:], hid_state[-1,:,:]), dim = 1)))# torch.cat convert to one dimension, self.fc apply a linear transformation to the data                  
         return outp, hidden  #We return the outputs and the hidden states
