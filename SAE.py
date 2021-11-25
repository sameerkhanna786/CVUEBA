from sklearn.preprocessing import MinMaxScaler
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

import math
import numpy as np

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error
import pandas as pd

"""
Custom implementation of a SAE model. Derived based on knowledge
and insight gleaned from Prof. Andrew Ng's notes on the subject:

https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
"""

class SAE:
    def __init__(self, mid_layer_size=1024, rho=0.05, beta=0.001, activation='relu', optimizer='adam', epochs=500, batch_size=128, scale = True, patience=5, lambda1=10e-3, lambda2=10e-3, verbose=1, folder_name = "SavedModels", retrain = False, save = True):
        self.mid_layer_size=mid_layer_size
        self.rho=rho  # sparse parameters
        self.beta=beta
        self.optimizer=optimizer
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.scale = scale
        self.patience = patience
        self.standScale = MinMaxScaler()

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.trained = False
        self.verbose = verbose

        self.autoencoder = None
        self.projectModel = None

        self.kld = tf.keras.losses.KLDivergence()

        self.folder_name = folder_name
        self.retrain = retrain
        self.save = save

    def build(self, input_dim):
        model = Sequential()
        model.add(Input(shape=(input_dim, )))
        model.add(Dense(self.mid_layer_size, activation=self.activation, kernel_regularizer=regularizers.l1(self.lambda1)))
        model.add(Dense(input_dim, activation=self.activation, kernel_regularizer=regularizers.l1(self.lambda2)))
        return model

    def encode(self,X):
        if self.projectModel is None:
            input_layer = Input(shape=(X.shape[1], ), name="ProjectionInput")
            model = input_layer
            for layer in self.autoencoder.layers:
                layer_size = 0
                if len(layer.input_shape) == 1:
                    layer_size = layer.input_shape[0][1]
                else:
                    layer_size = layer.input_shape[1]
                if layer_size != self.mid_layer_size:
                    model = layer(model)
                else:
                    break
            self.projectModel = Model(inputs=input_layer, outputs=model, name="ProjModel")
        if self.trained:
            return self.projectModel.predict(X)
        return self.projectModel(X)

    def loss(self, X_true, X_pred):
        H = self.encode(X_pred)
        rho_hat=tf.reduce_mean(H,axis=0)   #Average hidden layer over all data points in X, Page 14 in https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
        kl=self.kld(self.rho, rho_hat)
        mse = tf.keras.losses.mean_squared_error(X_true, X_pred)
        return self.beta*kl**2 + mse

    def kl_divergence(self, rho, rho_hat):
        return rho * tf.math.log(rho) - rho * tf.math.log(rho_hat) + (1 - rho) * tf.math.log(1 - rho) - (1 - rho) * tf.math.log(1 - rho_hat)

    def load(self):
        self.autoencoder = tf.keras.models.load_model("sae")

    def fit(self, X, y=None):
        if self.scale:
            self.standScale.fit(X)
            X = np.array(self.standScale.transform(X))

        input_dim = X.shape[1]

        try:
            self.autoencoder = tf.keras.models.load_model("sae", custom_objects={ 'loss': self.loss })
            if not self.retrain:
                self.trained = True
                return
        except Exception as e:
            self.autoencoder = self.build(input_dim)
            self.autoencoder.compile(optimizer=self.optimizer, 
                             loss=self.loss)

        #Reset projection model
        self.projectModel = None

        overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = self.patience, restore_best_weights=True)
        
        self.autoencoder.fit(X, X,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                use_multiprocessing=True,
                verbose=self.verbose,
                callbacks = [overfitCallback])
        if self.save:
            self.autoencoder.save("sae")
        self.trained = True

    def predict(self, X):
        if self.scale:
            X = np.array(self.standScale.transform(X))
        return self.encode(X)
