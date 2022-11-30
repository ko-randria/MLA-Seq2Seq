from tensorflow.keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from tensorflow.keras.layers import Bidirectional , Attention, Layer
from tensorflow.keras.activations import *
class Decoder(Layer):
  def __init__(self,embdim, intdim,outdim, enc_dimh, dec_dim_h, t_drop):
    self.emb=Embedding(intdim,enbdim)
    self.a=Attention(enc_dih,dec_dim_h)
    self.RNN=GRU(embdim+ enc_dimh))
    self.dropout=Dropout(t_drop)
    self.output=Dense(dec_dim_h,outdim)
    
    
  def decoder(self,Layer,enc_out,enc_hs,):
    emb=self.dropout(self.emb(Layer))
    
    #compute the attention weights :
    a=self.a(enchs[-1],enc_out)   #Last hiddden and all encoder outputs.
    
    
    
