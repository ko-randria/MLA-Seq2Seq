import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense , Input , SimpleRNN, LSTM , Embedding, Dropout, GRU
from keras.layers import Bidirectional , Attention, Layer
from keras.activations import *
import Encoder
import Decoder


class model_training:
    def __init__(self, param, model): #loss

        self.model = model
        self.loss_function = tf.keras.losses.MeanSquaredError() #CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=10e-6, rho=0.95) # SGD algorithm with Adadelta

        self.n = param[0] # 1000 - hidden layer
        self.m = param[1] # 620 - word embedding
        self.l = param[2] # 500 - maxeout hidden layour in the deep output
        self.n_p = param[3] # 1000 - The number of hidden units in the alignments model


    def convert_time(total_s):
        total_s = np.round(total_s / 60) * 60
        time_h = np.floor(total_s / 3600)
        total_s -= time_h * 3600
        time_m = np.floor(total_s / 60)
        total_s -= time_m * 60
        time_s = total_s
        return '%ih:%im:%.2fs'%(time_h,time_m,time_s)

    #@tf.function
    def train_step(self, data, labels):
        # Training step, this methods return the loss value for one step of learning on the training dataset
        with tf.GradientTape() as tape:
            # forward pass
            #predictions = self.model(inputs=[data, labels])
            predictions = self.model.predict([data, labels])
            labels = tf.cast(labels, tf.float32) 
            predictions = tf.cast(predictions, tf.float32) 
            #loss = self.loss_function(labels, predictions)
            loss = self.model.loss(labels, predictions)
            
        # computing the gradient
        gradient = tape.gradient(loss, self.model.trainable_variables)

        # retropropragation
        #self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return loss

    #@tf.function
    def evaluate_step(self, data, labels):
        with tf.GradientTape() as tape:
            # forward pass
            #predictions = self.model.call(data, labels)
            #predictions = self.model(inputs=[data, labels])
            predictions = self.model.predict([data, labels])
            #loss = self.loss_function(labels, predictions)
            loss = self.model.loss(labels, predictions)

        # Compute the gradients
        gradient = tape.gradient(loss, self.model.trainable_variables)

        # retropropagation
        #self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

    def training_loop(self, epochs, train_set, output, print_metric):
        average_time_step = []
        
        total_steps=epochs*np.shape(train_set)[0]
        for epoch in range(epochs):
            start_epoch=time.time()
            train_loss = 0

            for step in range(np.shape(train_set)[0]):
                start_step = time.time()
                train_loss += self.train_step(train_set[step], output[step])
                # evaluate_loss = self.evaluate_step()
                average_time_step.append(time.time()-start_step)
            

            if epoch%int(epochs/10) == 0:
                #print('Epoch %d, Loss %f' % (epoch, float(train_loss)))
                if print_metric: # on peut choisir si on affiche les metrics

                    print("\rEpoch %i/%i - Step %i/%i - Loss : %s%.3f (remaining time : %s)"
                        %(epoch+1, epochs, step+1, train_set.__len__(), ' '*(4-len(str(int(np.round(train_loss))))), train_loss, self.convert_time(
                            total_s=(total_steps - epoch*train_set.__len__() - step) * np.mean(average_time_step))), end='')
        return self.model