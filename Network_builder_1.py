# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt
import cv2
import os
from image_preprocess import generate_data
import math

def create_cnn(inputs, filters):
    if filters!=1:
        cnn = layers.SeparableConv2D(filters, kernel_size=(3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                                     depth_multiplier=1,
                                     use_bias=True, 
                                     depthwise_initializer='glorot_uniform', 
                                     pointwise_initializer='glorot_uniform',
                                     bias_initializer='zeros')(inputs)
        cnn = layers.LeakyReLU(alpha=0.1)(cnn)
        
        batch_layer = layers.BatchNormalization()(cnn)
        cnn_dropout = layers.Dropout(0.2)(batch_layer)
        print('Cnn', cnn.shape)
        return cnn, batch_layer, cnn_dropout
    else:
        pool = layers.AveragePooling2D(pool_size=(2,2),strides=2)(inputs)
        print('Pool', pool.shape)
        return pool


def create_dense(inputs, units):
    dense = layers.Dense(units, kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    dense = layers.LeakyReLU(alpha=0.1)(dense)
    dense_dropout = layers.Dropout(0.2)(dense)
    dense_batch = layers.BatchNormalization()(dense_dropout)
    print('Dense', dense.shape)
    return dense, dense_dropout, dense_batch


def backbone(num,feature_len,inputs_cnn):
   
    cnn_num_layers = num[0]
    filters = num[2: 2+cnn_num_layers]
    cnn_name = list(np.zeros((cnn_num_layers,)))
    batch_layer_name = list(np.zeros((cnn_num_layers,)))
    cnn_dropout_name = list(np.zeros((cnn_num_layers,)))
    pool_name = list(np.zeros((cnn_num_layers,)))
    
    cnn_dense_num_layers = num[1]
    cnn_dense_units = num[2+cnn_num_layers: 2+cnn_num_layers+cnn_dense_num_layers]
    cnn_dense_name = list(np.zeros((cnn_dense_num_layers,)))
    cnn_dense_dropout_name = list(np.zeros((cnn_dense_num_layers,)))
    cnn_dense_batch_name = list(np.zeros((cnn_dense_num_layers,)))
    

    for i in range(cnn_num_layers):
        if i == 0:
            inputs = inputs_cnn
        else:
            if filters[i-1]==1:
                inputs = pool_name[i-1]
            else:
                inputs = cnn_dropout_name[i-1]
            
        if filters[i]!=1:
            cnn_name[i], batch_layer_name[i], cnn_dropout_name[i] = create_cnn(inputs, filters[i])
        if filters[i]==1:
            pool_name[i] = create_cnn(inputs, filters[i])
            
    if filters[cnn_num_layers-1]==1:
        flatten = layers.Flatten()(pool_name[cnn_num_layers-1])
    if filters[cnn_num_layers-1]!=1:
        flatten = layers.Flatten()(cnn_dropout_name[cnn_num_layers-1])
    

    for i in range(cnn_dense_num_layers):
        if i == 0:
            inputs = flatten
        else:
            inputs = cnn_dense_batch_name[i-1]
        cnn_dense_name[i], cnn_dense_dropout_name[i], cnn_dense_batch_name[i] = create_dense(inputs, cnn_dense_units[i])
 
    feature = layers.Dense(feature_len)(cnn_dense_batch_name[cnn_dense_num_layers-1])
    
    return feature

def AFF(x_1,x_2,r,feature_len):
    
    channels= feature_len
    inter_channels = int(channels // r)
    x= layers.add([x_1, x_2])

    local_att = layers.SeparableConv2D(inter_channels, kernel_size=(1,1), strides=(1,1), padding='same', 
                                       depth_multiplier=1,
                                       use_bias=True, 
                                       depthwise_initializer='glorot_uniform', 
                                       pointwise_initializer='glorot_uniform',
                                       bias_initializer='zeros')(x)
    local_att = layers.BatchNormalization()(local_att)
    local_att = layers.ReLU()(local_att)
    local_att = layers.SeparableConv2D(channels, kernel_size=(1,1), strides=(1,1), padding='same',
                                       depth_multiplier=1,
                                       use_bias=True, 
                                       depthwise_initializer='glorot_uniform', 
                                       pointwise_initializer='glorot_uniform',
                                       bias_initializer='zeros')(local_att)
    local_att = layers.BatchNormalization()(local_att)
 
    

    global_att = layers.SeparableConv2D(inter_channels, kernel_size=(1,1), strides=(1,1), padding='same',
                                        depth_multiplier=1,
                                        use_bias=True, 
                                        depthwise_initializer='glorot_uniform', 
                                        pointwise_initializer='glorot_uniform',
                                        bias_initializer='zeros')(x)
    global_att = layers.BatchNormalization()(global_att)
    global_att = layers.ReLU()(global_att)
    global_att = layers.SeparableConv2D(channels, kernel_size=(1,1), strides=(1,1), padding='same',
                                        depth_multiplier=1,
                                        use_bias=True, 
                                        depthwise_initializer='glorot_uniform', 
                                        pointwise_initializer='glorot_uniform',
                                        bias_initializer='zeros')(global_att)
    global_att = layers.BatchNormalization()(global_att)
 
    xlg= layers.add([local_att, global_att])
    wei = tf.keras.activations.sigmoid(xlg)
    xo =  x_1 * wei +  x_2 * (1 - wei)
    return xo

   
def classify(path_train,path_val,num_1,num_2,feature_len):
    
    height = 299
    width  = 299
    channel_1 = 3
    channel_2 = 1
    batch_size =30
    inputs_cnn_1 = layers.Input(shape=(height, width, channel_1))
    inputs_cnn_2 = layers.Input(shape=(height, width, channel_2))
    
    feature_1 = backbone(num_1,feature_len,inputs_cnn_1)
    feature_2 = backbone(num_2,feature_len,inputs_cnn_2)
    feature_1 = tf.expand_dims(feature_1, axis=1)
    feature_1 = tf.expand_dims(feature_1, axis=1)
    feature_2 = tf.expand_dims(feature_2, axis=1)
    feature_2 = tf.expand_dims(feature_2, axis=1)
    FF=AFF(feature_1,feature_2,5,feature_len)
    
    outputs_cnn = layers.Dense(2,activation='softmax')(FF)
    outputs_cnn = layers.Flatten()(outputs_cnn)
   
   
    CNN_model = keras.Model([inputs_cnn_1,inputs_cnn_2], outputs_cnn)
    CNN_model.compile(optimizer=keras.optimizers.Adam(lr=2e-3),
                      # loss=keras.losses.CategoricalCrossentropy(),  
                      # loss=keras.losses.SparseCategoricalCrossentropy(),
                      loss=['categorical_crossentropy'],
                      metrics=['categorical_accuracy'])

    history = CNN_model.fit_generator(generate_data(path_train[0],path_train[1],height,width,batch_size=batch_size,num=[0]),
                                      steps_per_epoch = math.ceil(25000 / batch_size), 
                                      epochs=4)
    s=0
    results=[]
    fitness=[]
    for i in generate_data(path_val[0],path_val[1],height,width,batch_size=250,num=[0]):
        s=s+1
        if s<10:
            results.append(CNN_model.evaluate(i[0],i[1],verbose=0)[1])
        if s>=10:
            break

    means=np.mean(results)
    std=np.std(results)
    
    fitness.append(means)
    fitness.append(std)
    
    return np.array(fitness)
    






