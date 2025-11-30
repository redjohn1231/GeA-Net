# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import clear_session
import deep_learning as project
from tensorflow.keras.models import load_model
from image_preprocess import generate_data
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score,auc
from keras.callbacks import ReduceLROnPlateau

from keras import metrics
from Setting import dataset

import math

path_train_fake = dataset.path_train_0
path_train_real = dataset.path_train_1
path_train = [path_train_fake,path_train_real]

path_val_fake = dataset.path_val_0
path_val_real = dataset.path_val_1
path_val = [path_val_fake,path_val_real]


# Training configuration dictionary (example template)
train_config = {
    "height": 299,          # Input image height 
    "width": 299,           # Input image width 
    "channel_1": None,       # Number of channels for branch 1 
    "channel_2": None,       # Number of channels for branch 2 
    "batch_size": None,      # Training batch size
    "learning_rate": None,   # Learning rate for optimizer 
    "epochs": None,          # Number of training epochs
    "steps_per_epoch": None, # Number of steps per epoch during training
    "val_steps": None,       # Number of validation steps per validation round
    "val_batch_size": None,  # Validation batch size
}

def get_fitness(x_1,x_2,s): 
    return project.classify(s,path_train,path_val,num_1=x_1,num_2=x_2,feature_len=50,train_config=train_config)

def train(pop,s):
    for i in range(1):
        
        pop_list_1 = list(pop[0][i])
        pop_list_2 = list(pop[1][i])
    
        for j, each in enumerate(pop_list_1):
            if each == 0.0:
                pop_list_1 = pop_list_1[:j]
        for k, each in enumerate(pop_list_1):
                each_int = int(each)
                pop_list_1[k] = each_int
            
        for j, each in enumerate(pop_list_2):
            if each == 0.0:
                pop_list_2 = pop_list_2[:j]
        for k, each in enumerate(pop_list_2):
                each_int = int(each)
                pop_list_2[k] = each_int    
       
        #train
        get_fitness(pop_list_1,pop_list_2,s)
        clear_session()
    
pop_1=np.array([[12, 1, 37, 51, 1, 1, 187, 211, 209, 1, 402, 1,   166, 419, 32]]) 
pop_2=np.array([[12, 1, 38, 1,  1, 1, 174, 202, 173, 168, 1, 228, 361, 165, 35]]) 
pop=np.array([pop_1,pop_2])
train(pop,"GeA-Net")





    



