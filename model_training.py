from tensorflow.keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from tensorflow.keras.layers import Bidirectional , Attention, Layer
#from tensorflow.keras.activations import tanh
from tensorflow.keras.activations import *
import time
import numpy as np
import tensorflow as tf
import Encoder
import Decoder

# je suis encore en train de changer des choses niveau training-loop mais je fais une pause pour finir son une bonne fois pour toute
# je vous upload ça au cas ou en attendant

class model_training:
    def __init__(self, param, model, loss):

        # this is too create model in another function
        # self.encoder = Encoder(layer)
        # self.decoder = Decoder(layer)
        # self.attention = Attention()

        self.model = model
        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.n = 1000 # hidden layer
        self.m = 620 # word embedding
        self.l = 500 # maxeout hidden layour in the deep output
        self.n_p = 1000 # The number of hidden units in the alignments model

        # We initialize the weight matrices
        mu, sigma = 0, 0.001
        self.Ua = np.random.normal(mu, sigma ,size=(self.n,2*self.n))
        self.Wa = np.random.normal(mu, sigma ,size=(self.n_p,self.n))
        # I don't know if we have to initialize the different weight matrices for each step
        # like they said in the report
        # if we had to, we need to find how to initialize those following and how to force encoder and decoder to use them:
    # U =
    # Uz =
    # Ur =

    # # ???
    # U_f =
    # Uz_f =
    # Ur_f =

    # U_b =
    # Uz_b =
    # Ur_b =
    # # ???
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=10e-6, rho=0.95) # SGD algorithm with Adadelta

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            # forward pass
            predictions = self.model(data)
            loss = self.loss_function(labels, predictions)
        # calcul des gradients
        gradient = tape.gradient(loss, self.model.trainable_variables)
        # retropropagation
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

    def evaluate_step(self):
        pass

    def training_loop(self, epochs, train_set, print_metric):
        total_steps=epochs*np.shape(train_set)[0]
        for epoch in range(epochs):
            start_epoch=time.time()


            for step in range(len(train_set)):
                train_loss = self.train_step()
                evaluate_loss = self.evaluate_step()
                # loss=
                # loss+=loss
                
            

            if epoch%int(epochs/10) == 0:
                print('Epoch %d, Loss %f' % (epoch, float(train_loss)))
                if print_metric: # on peut choisir si on affiche les metrics


            """       print("\rEpoch %i/%i - Step %i/%i - Loss : %s%.3f (remaining time : %s)"
                    %(epoch+1, epochs, step+1, train_set.__len__(), ' '*(4-len(str(int(np.round(loss_value))))), loss_value, convert_time(
                        total_s=(total_steps - epoch*train_set.__len__() - step) * np.mean(average_time_step))), end='') """
        return Model