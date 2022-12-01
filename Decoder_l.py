class decoder (Layer) :

    def __init__(self,max_hid_lay= 500,  size_out, size_emb, enc_dimh, dec_dimh, t_drop, attention, iterat): 
        
        super(decoder, self).__init__()
        self.attention = attention_trad
        self.embedding = Embedding(size_out, size_emb) #Embedding matrix of the target word
        self.rnn = GRU (enc_dimh)
        self.dropout = Dropout(t_drop)
        
    def build (self, inp_shape = dec_hdim) : #State of the layer (weights)
            w_init = tf.keras.initializers.Zeros() 
            #Weight matrices 
            self.W0 = tf.Variable (initial_value=w_init(shape=([size_out,max_hid_lay]), self.units),) trainable=True)#shape : n, with n : number of hidden states, max_hid_lay = 500, with max_hid_lay : size of the maxout hidden layer in the deep output, symbole l in the article 
            self.U0 = tf.Variable (initial_value = w_init(shape=([2*max_hid_lay,enc_dimh] ,self.units),) trainable=True)
            self.V0 = tf.Variable (initial_value = w_init(shape=([2*max_hid_lay,size_emb] ,self.units),) trainable=True)
            self.C0 = tf.Variable (initial_value = w_init(shape=([2*max_hid_lay,2*enc_dimh] ,self.units),) trainable=True)
 
    def call (self, entr, hidden, outp_enc ) : 
       emb =  self.embedding(entr)
       emb = dropout(emb)
       attr = self.attention(hidden, outp_enc ) #We compute the attention between this two terms 
       outp, hidden = self.rnn(emb)
       t = self.U0*hidden+self.V0*emb*out+self.C0*attr
       ti = np.max(t[2*iterat-1:2*iterat])
        
       prediction = np.exp(outp.T*self.W0*ti)  

       return prediction, hidden #We send the predicted word yi, and new values of the hidden states 
    
