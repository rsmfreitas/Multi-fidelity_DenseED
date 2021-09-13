# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:20:30 2021

@author: rodol
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
from tensorflow.keras import Model, layers
import pandas as pd
# Helper libraries
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

tf.random.set_seed(123)
np.random.seed(123)

class NN_model:
    def __init__(self,beta=1e-4):
        self.beta = beta
    
    def linear(self,inputs):
        x = tf.keras.layers.Conv2DTranspose(1, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=tf.keras.regularizers.l2(self.beta))(inputs)
        x = tf.math.sigmoid(x)    
        model = tf.keras.Model(inputs,x)
        return model
        

class Dense_ED: 
    def __init__(self, L=5, K=24, beta=1e-4):
        self.L = L
        self.K = K
        self.beta = beta
    
    def encoder(self,x):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(x.shape[-1], kernel_size=(3,3), strides=(2,2), padding='same',kernel_regularizer=tf.keras.regularizers.l2(self.beta))(x)
        return x
    
    # decoder layer
    def decoder(self,x):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = tf.keras.layers.Conv2D(int(x.shape[-1]), kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=tf.keras.regularizers.l2(self.beta))(x)
        return x
    
    def dense_block(self,x):
        for idx in range(self.L):
            out = tf.keras.layers.BatchNormalization()(x)
            out = tf.nn.relu(out)
            out = tf.keras.layers.Conv2D(self.K,kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=tf.keras.regularizers.l2(self.beta))(out)
            x   = tf.keras.layers.concatenate([x, out],axis=-1)
        return x
    
    # Encoder-Decoder model
    def encoder_decoder_model(self,inputs):
        x = tf.keras.layers.Conv2D(48, kernel_size=(5,5), strides=(2,2), padding='same',kernel_regularizer=tf.keras.regularizers.l2(self.beta))(inputs)
        x = self.dense_block(x)
        x = self.encoder(x)
        x = self.dense_block(x)
        x = self.encoder(x)
        x = self.dense_block(x)
        x = self.decoder(x)
        x = self.dense_block(x)
        x = self.decoder(x)
        x = self.dense_block(x)
        x = tf.keras.layers.Conv2DTranspose(1, kernel_size=(5,5), strides=(2,2), padding='same',kernel_regularizer=tf.keras.regularizers.l2(self.beta))(x)
        x = tf.math.sigmoid(x)    
        model = tf.keras.Model(inputs,x)
        return model
    
        

# Metrics
def relative_error(y_true, y_pred):
    r_error = tf.math.reduce_mean(tf.math.sqrt(tf.math.square(y_true-y_pred))/tf.math.sqrt(tf.math.square(y_true)))
    return r_error

def rmse_metric(y_true, y_pred):
    rmse = tf.sqrt(tf.math.reduce_mean(tf.math.square(y_true-y_pred)))
    return rmse

def mse(y_true, y_pred):
    mse = tf.math.reduce_mean(tf.math.square(y_true-y_pred))
    return mse

# l2 mean relative error
def l2_mre(y_true, y_pred):
    mre = tf.math.reduce_mean(tf.math.square((y_true-y_pred)+1e-8)/tf.math.square(y_true+1e-8))
    return mre

# Coefficient of determination R-squared
def r_squared(y_true,y_pred):
    SS_res =  tf.reduce_sum(tf.square( y_true-y_pred ))
    SS_tot = tf.reduce_sum(tf.square( y_true - tf.reduce_mean(y_true) ) )
    r2 = 1.0 - SS_res/SS_tot 
    return r2

# Optimization process. Inputs: real image and noise.
def run_optimization(x_L,y_L,x_H,y_H):
    
    # Compute the gradients of the Low-fidelity model
    with tf.GradientTape() as g_LF:
        y_pred_LF = model_LF(x_L)
        loss_LF   = mse(y_L, y_pred_LF)
        y_pred_HF = model_HF_NL(x_H) + model_HF_L(x_H)
        loss_HF   = mse(y_H, y_pred_HF)
        loss      = loss_LF + loss_HF 
    
    gradients_LF    = g_LF.gradient(loss,  model_LF.trainable_variables)
    
    # Compute the gradients of the High-fidelity model(Linear model)
    with tf.GradientTape() as g_HF_L:
        y_pred_LF = model_LF(x_L)
        loss_LF   = mse(y_L, y_pred_LF)
        y_pred_HF = model_HF_NL(x_H) + model_HF_L(x_H)
        loss_HF   = mse(y_H, y_pred_HF)
        loss      = loss_LF + loss_HF 
    
    gradients_HF_L  = g_HF_L.gradient(loss,  model_HF_L.trainable_variables)
    
    # Compute the gradients of the High-fidelity model(Non-linear model)
    with tf.GradientTape() as g_HF_NL:
        y_pred_LF = model_LF(x_L)
        loss_LF   = mse(y_L, y_pred_LF)
        y_pred_HF = model_HF_NL(x_H) + model_HF_L(x_H)
        loss_HF   = mse(y_H, y_pred_HF)
        loss      = loss_LF + loss_HF 
    
    gradients_HF_NL  = g_HF_NL.gradient(loss,  model_HF_NL.trainable_variables)
    
    # Optimize the train variables
    optimizer.apply_gradients(zip(gradients_LF,  model_LF.trainable_variables))
    optimizer.apply_gradients(zip(gradients_HF_L,  model_HF_L.trainable_variables))
    optimizer.apply_gradients(zip(gradients_HF_NL,  model_HF_NL.trainable_variables))
     
    return loss_LF, loss_HF, loss

# Fetches a mini-batch of data
def fetch_minibatch(X, Y, N_batch):
    N = X.shape[0] 
    idx = np.random.choice(N, N_batch, replace=False) 
    X_batch = X[idx,:,:,:] 
    Y_batch = Y[idx,:,:,:] 
    return X_batch, Y_batch

def run_train(x_L,y_L,x_H,y_H, batch_size, N_iter):
  save_loss = []
  for it in range(1,N_iter+1):  
    start = time.time()
    batch_x_L,batch_y_L = fetch_minibatch(x_L, y_L, batch_size)
    batch_x_H,batch_y_H = fetch_minibatch(x_H, y_H, batch_size)
    
    loss_low, loss_high, loss = run_optimization(batch_x_L,batch_y_L,batch_x_H,batch_y_H)
        
    # lOG EVERY 10 Iterations
    if it % 100 == 0:
          print("Iteration: %i, loss_L: %.4e, loss_H: %.4e, loss: %.4e, time(s): %f" % (it, loss_low,loss_high, loss, time.time()-start))
    
    # Save the losses
    save_loss.append(loss)
    
  
  return save_loss

if __name__=="__main__":
    
    n_MC              = 1500
    'Low-fidelity data'
    # Training data set
    v_L               = np.load('velocity_6layers_Low.npy')  # velocity field
    Im_L              = np.load('Image_6layers_Low.npy') # RTM Image (the image to ranging in I~[0,1])   
    
    N_L               = 1000
    N_H               = 100
    # Split in train and test sets       
    v_L_train         = v_L[:N_L,:,:]
    v_L_test          = v_L[1000:,:,:] 
    Im_L_train        = Im_L[:N_L,:,:]
    Im_L_test         = Im_L[1000:,:,:]
    
    # Normalize the velocity field
    mean            = v_L_train.mean(axis=0)
    std             = v_L_train.std(axis=0)
    v_L_train         = (v_L_train - mean) / std
    v_L_test          = (v_L_test - mean) / std
    
    # Convert to float32
    v_L_train, v_L_test = np.array(v_L_train, np.float32), np.array(v_L_test, np.float32)
    Im_L_train, Im_L_test = np.array(Im_L_train, np.float32), np.array(Im_L_test, np.float32)
    
    # Reshape to match picture format [Height x Width x Channel]
    v_L_train         = v_L_train.reshape(v_L_train.shape[0],v_L_train.shape[1],v_L_train.shape[2],1)
    Im_L_train        = Im_L_train.reshape(Im_L_train.shape[0],Im_L_train.shape[1],Im_L_train.shape[2],1)
    
    v_L_test          = v_L_test.reshape(v_L_test.shape[0],v_L_test.shape[1],v_L_test.shape[2],1)
    Im_L_test         = Im_L_test.reshape(Im_L_test.shape[0],Im_L_test.shape[1],Im_L_test.shape[2],1)
    
     # low-fidelity training data at x_H
    Im_L_HF = np.zeros((N_H,144,144))
    for i in range(N_H): 
        Im_L_HF[i,:,:]  = cv2.resize(Im_L[i,:,:],(144,144))
    
    Im_L_HF        = Im_L_HF.reshape(Im_L_HF.shape[0],Im_L_HF.shape[1],Im_L_HF.shape[2],1)
    
    'High-fidelity data'
    # Training data set
    v_H               = np.load('velocity_6layers_High.npy')  # velocity field
    Im_H              = np.load('Image_6layers_High.npy') # RTM Image (the image to ranging in I~[0,1])   
    
    # Split in train and test sets       
    v_H_train         = v_H[:N_H,:,:]
    v_H_test          = v_H[1000:,:,:] 
    Im_H_train        = Im_H[:N_H,:,:]
    Im_H_test         = Im_H[1000:,:,:]
    
    # Normalize the velocity field
    mean            = v_H_train.mean(axis=0)
    std             = v_H_train.std(axis=0)
    v_H_train       = (v_H_train - mean) / std
    v_H_test        = (v_H_test - mean) / std
    
    # Convert to float32
    v_H_train, v_H_test = np.array(v_H_train, np.float32), np.array(v_H_test, np.float32)
    Im_H_train, Im_H_test = np.array(Im_H_train, np.float32), np.array(Im_H_test, np.float32)
    
    # Reshape to match picture format [Height x Width x Channel]
    v_H_train         = v_H_train.reshape(v_H_train.shape[0],v_H_train.shape[1],v_H_train.shape[2],1)
    Im_H_train        = Im_H_train.reshape(Im_H_train.shape[0],Im_H_train.shape[1],Im_H_train.shape[2],1)
    
    v_H_test          = v_H_test.reshape(v_H_test.shape[0],v_H_test.shape[1],v_H_test.shape[2],1)
    Im_H_test         = Im_H_test.reshape(Im_H_test.shape[0],Im_H_test.shape[1],Im_H_test.shape[2],1)
    
    # Concatenat (v_HF, Im_LF(v_HF))
    v_H_train         = np.concatenate([v_H_train,Im_L_HF],axis=-1) 
    
    ################### Hyperparameters
    # Stochastic gradient descent optimizer.
    N_batch         = 32
    training_iter   = 20000
    K               = 16    # growth rate
    L               = 4     # number of layers in dense block
    learning_rate   = 1e-4 # learning rate
    optimizer       = tf.keras.optimizers.Adam(learning_rate)
  
    ############# Create the model
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    inputs_LF       = tf.keras.Input(shape=(v_L_train.shape[1],v_L_train.shape[2],1),dtype='float32')
    inputs_HF       = tf.keras.Input(shape=(v_H_train.shape[1],v_H_train.shape[2],2),dtype='float32')
    
    # models
    model_LF        = Dense_ED(L,K,beta=0.).encoder_decoder_model(inputs_LF)
    model_HF_NL     = Dense_ED(L,K,beta=0.01).encoder_decoder_model(inputs_HF)
    model_HF_L      = NN_model(beta=0.).linear(inputs_HF)     
    
    # ############################### Train the model #########################################
    start_time = time.time()
    loss = run_train(v_L_train,Im_L_train,v_H_train,Im_H_train, N_batch, training_iter)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    # Save the entire model to a HDF5 file.
    #model.save('saved_model_1000/surrogate_RTM2D_marmousi') 
    
    # Test model on validation set.
    start_time = time.time()
    aux_pred   = model_LF.predict(v_L_test)
    aux_pred   = aux_pred.reshape(aux_pred.shape[0],aux_pred.shape[1],aux_pred.shape[2])
    Im_pred_LF = np.zeros((500,144,144))
    for i in range(500): 
        Im_pred_LF[i,:,:]  = cv2.resize(aux_pred[i,:,:],(144,144))
    Im_pred_LF = Im_pred_LF.reshape(Im_pred_LF.shape[0],Im_pred_LF.shape[1],Im_pred_LF.shape[2],1)
    v_H_test   = np.concatenate([v_H_test,Im_pred_LF],axis=-1)
    Im_pred_HF = model_HF_NL.predict(v_H_test) + model_HF_L.predict(v_H_test)  
    elapsed    = time.time() - start_time
    print('Prediction time: %.4f' % (elapsed))
    Im_pred_HF = Im_pred_HF.reshape(Im_pred_HF.shape[0],Im_pred_HF.shape[1],Im_pred_HF.shape[2])
    
    #np.save('results_model_1000/Im_predictions_test_samples',Im_pred)
    
    Im_H_test = Im_H_test.reshape(Im_pred_HF.shape[0],Im_pred_HF.shape[1],Im_pred_HF.shape[2])
    
     # Coefficient of determination R-squared
    R2 = r_squared(Im_H_test,Im_pred_HF)
    rmse = rmse_metric(Im_H_test,Im_pred_HF)
    SSIM = tf.image.ssim(Im_H_test, Im_pred_HF, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    print('Coefficient of determination R-squared: %.4f' % (R2))
    print('Root mean square error: %.4f' % (rmse))
    print('Structural Similarity Image: %.4f' % (SSIM))
    
    n           = len(Im_H_test)
    i           = np.random.randint(n,size=3,)
    
      
    plt.figure()
    plt.imshow(Im_H_test[i[0]].reshape(Im_H_test.shape[1],Im_H_test.shape[2]), origin='upper',extent=[0,1000,1000,0], cmap='gray')
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Length [m]',fontsize=12)
    plt.ylabel('Depth [m]',fontsize=12)
    plt.savefig('Im_6layers_test.png', bbox_inches='tight')

    plt.figure()
    plt.imshow(Im_pred_HF[i[0],:,:], origin='upper',extent=[0,1000,1000,0], cmap='gray')
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Length [m]',fontsize=12)
    plt.ylabel('Depth [m]',fontsize=12)
    plt.savefig('Im_6layers_pred.png', bbox_inches='tight')
    
    error_abs = (np.abs(Im_H_test[i[0]].reshape(Im_H_test.shape[1],Im_H_test.shape[2])-Im_pred_HF[i[0],:,:]) + 1e-6) / (Im_H_test[i[0]].reshape(Im_H_test.shape[1],Im_H_test.shape[2]) + 1e-6)
    
    
    plt.figure()
    plt.imshow(error_abs,origin='upper',extent=[0,9200,3500,0],cmap='jet',vmax= 0.02)
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Length [m]',fontsize=12)
    plt.ylabel('Depth [m]',fontsize=12)
    plt.savefig('relative_error.png',bbox_inches='tight')
    
    