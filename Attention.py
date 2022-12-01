class attention_trad (Layer) :
    def __init__(self, enc_outdim, dec_hdim) :    #We need the hidden state of the decoder and the encoder 

        super(attention_trad, self).__init__()
        
    def build (self, inp_shape = dec_hdim) : #State of the layer (weights)
        w_init = tf.keras.initializers.Zeros() 
        #Weight matrices 
        self.Va = tf.Variable (initial_value=w_init(shape=(inp_shape[-1], self.units),) trainable=True)#shape : n, with n : number of hidden states 
        self.Wa = tf.Variable (initial_value = w_init(shape=([inp_shape[-1],inp_shape[-1]],self.units),) trainable=True) #shape : n*n
        self.Ua = tf.Variable (initial_value = w_init(shape=([inp_shape[-1],inp_shape[-1]],self.units),)trainable=True) #shape : n*2n
        
    def call (self, outp_enc, hidden_dec ) : # hidden state of the decoder, outp_enc = outputs of the encoder
        #We compute the attention : 
        a  = self.Va.T*tanh(self.Wa*hidden_dec[-1]+self.Ua*outp_enc )  #hidden_dec[-1] : decoder hidden state of i-1,
        alpha = tf.nn.softmax(a) #We normalize the output probabilities
         
        c = np.sum(alpha*hidden_dec) #context 
        return c
 
