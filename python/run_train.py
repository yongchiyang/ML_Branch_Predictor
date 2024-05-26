# This code is adapt from https://github.com/zzmicer/Branch-Prediction-Decision-Trees/tree/master
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,LeakyReLU, Conv1D, Flatten, MaxPooling1D, BatchNormalization

from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

NUM_SAMPLES=10

def create_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(NUM_SAMPLES,)))
    model.add(BatchNormalization())
    model.add(Dense(NUM_SAMPLES))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
              
    model.add(Dense(NUM_SAMPLES//2))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
              
    model.add(Dense(NUM_SAMPLES//2))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
    
    model.add(Dense(1,activation="sigmoid"))
    
    return model             

def create_cnn_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(NUM_SAMPLES,1)))
    model.add(Conv1D(filters=3,kernel_size=13,strides=1,groups=1))
    model.add(Conv1D(filters=3,kernel_size=7,strides=1,groups=1))
    model.add(Conv1D(filters=3,kernel_size=5,strides=1,groups=1))
    model.add(Conv1D(filters=3,kernel_size=3,strides=1,groups=1))
    model.add(Flatten())
    
    model.add(Dense(NUM_SAMPLES))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
    model.add(Dense(NUM_SAMPLES//2))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
    
    model.add(Dense(1,activation="sigmoid"))
    
    return model    

# model compile
model = create_model()
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['acc'])

#model = tf.keras.models.load_model('my_model.keras')

# training
traces = [("SHORT_MOBILE",2)]

for trace,iter in traces:
    for i in range(iter+1):

        csv_name = os.path.join("../data/train/","{trace}-split0000{i}".format(trace,i))
        data = pd.read_csv(csv_name,header=None)
        print("read data {csv_name} done.".format(csv_name))

        X_train = np.array(data.iloc[:,:-1])
        y_train = np.array(data.iloc[:,-1])
        y_train = y_train[:,np.newaxis]
        data = None

        model.fit(X_train,y_train,epochs=1,batch_size=32)
        model.save('{trace}.keras'.format(trace))

