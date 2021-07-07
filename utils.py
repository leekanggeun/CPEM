import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import scipy.io
import numpy as np
import scipy
import os
import hdf5storage
import tensorflow_addons as tfa
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

#Switchable Normalization
class SN(layers.Layer):
    def __init__(self, activation):
        super(SN, self).__init__()
        self.activation = activation
    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.eps = 1e-5
        self.momentum = 0.997
        self.gamma = tf.Variable(tf.ones([1,1,1,self.ch], dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(tf.zeros([1,1,1,self.ch], dtype=tf.float32), trainable=True)
        self.mean_weight = tf.nn.softmax(tf.Variable([1,1,1], dtype=tf.float32, trainable=True))
        self.var_weight = tf.nn.softmax(tf.Variable([1,1,1], dtype=tf.float32, trainable=True))
        self.moving_average_mean = tf.Variable(tf.zeros([1,1,1,input_shape[3]]), dtype=tf.float32, trainable=False)
        self.moving_average_var = tf.Variable(tf.zeros([1,1,1,input_shape[3]]), dtype=tf.float32, trainable=False)

        self.act = layers.Activation(activation=self.activation)

    def call(self, input, act=True, training=True):
        i_mean, i_var = tf.nn.moments(input, [1], keepdims=True) # N, 1, 1, C
        l_mean, l_var = tf.nn.moments(input, [1,2], keepdims=True) # N, 1, 1, 1
        if training:
            b_mean, b_var = tf.nn.moments(input, [0,1], keepdims=True)
            keras.backend.moving_average_update(self.moving_average_mean, b_mean, self.momentum)
            keras.backend.moving_average_update(self.moving_average_var, b_var, self.momentum)
        else:
            b_mean = self.moving_average_mean
            b_var = self.moving_average_var

        mean = self.mean_weight[0]*b_mean + self.mean_weight[1]*i_mean + self.mean_weight[2]*l_mean
        var = self.var_weight[0]*b_var + self.var_weight[1]*i_var + self.var_weight[2]*l_var
        x = (input - mean)/(tf.sqrt(var+self.eps))
        return self.act(x*self.gamma+self.beta) if act else x*self.gamma + self.beta

#Instance Normalization
class IN(layers.Layer):
    def __init__(self, activation):
        super(IN, self).__init__()
        self.eps = 1e-7
        self.activation = layers.Activation(activation=activation)
    def call(self, input, act=True, training=True):
        i_mean, i_var = tf.nn.moments(input, [1], keepdims=True) # N, 1, 1, C
        x = (input-i_mean)/(tf.sqrt(i_var+self.eps))
        return self.activation(x) if act else x



class CPEM_DNN(keras.Model):
    def __init__(self):
        super(CPEM_DNN, self).__init__()
        self.nparameters = 4096
        self.out_channel = 31
        self.rate = 0.4
        self.h1 = layers.Dense(self.nparameters, activation=tf.nn.leaky_relu)
        self.h2 = layers.Dense(self.nparameters, activation=tf.nn.leaky_relu)
        self.h3 = layers.Dense(self.nparameters, activation=tf.nn.leaky_relu)
        self.h4 = layers.Dense(self.out_channel, activation=None)

        

    def call(self, inputs, training=True):
        h = self.h1(inputs)
        h = layers.Dropout(self.rate)(h, training=training)
        h = self.h2(h)
        h = layers.Dropout(self.rate)(h, training=training)
        h = self.h3(h)
        h = layers.Dropout(self.rate)(h, training=training)
        h = self.h4(h)
        return tf.nn.softmax(h)
    
    def predict(self, inputs):
        return self(inputs, training=False)

class LRRecorder(keras.callbacks.Callback):
    """Record current learning rate. """
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        print("The current learning rate is {}".format(lr.numpy()))
