# This code is adapt from https://github.com/zzmicer/Branch-Prediction-Decision-Trees/tree/master
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,LeakyReLU

from tensorflow.keras.optimizers import Adam

NUM_SAMPLES=40

def create_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(NUM_SAMPLES,)))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(1,activation="sigmoid"))

    return model

# training traces
traces = [("SHORT_SERVER",1018),("SHORT_MOBILE",214),("LONG_SERVER",95),("LONG_MOBILE",126)]

for trace,iter in traces:
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['acc'])

    for i in range(0,iter+1):

        csv_name = os.path.join("../../data/generated/{}.train/".format(trace),"{}-fin-{}".format(trace,i))
        data = pd.read_csv(csv_name,header=None)
        print("read data {} done.".format(csv_name))

        X_train = np.array(data.iloc[:,:-1])
        y_train = np.array(data.iloc[:,-1])
        y_train = y_train[:,np.newaxis]
        data = None

        model.fit(X_train,y_train,epochs=1,batch_size=32)
        X_train = None
        model.save('{}.keras'.format(trace))

