import os
import numpy as np
import pandas as pd
import tensorflow as tf

def mispredict_results(trace_name,y_pred,y_test,y_cnt,len):
    total, mispred = 0,0
    for i in range(len):
        total += y_cnt[i]
        if(y_pred[i] != y_test[i]): mispred += y_cnt[i]
    
    with open("mispredict.txt",'a') as file:
        file.write("{0},{1},{2}\n",trace_name,total,mispred)

# load model
model = tf.keras.models.load_model('my_model.keras')

traces = [("SHORT_MOBILE",10)]

for trace,iter in traces:
    for i in range(1,iter+1):
        csv_name = os.path.join("../data/test/","{}-{}.bt9.trace".format(trace,i))
        data = pd.read_csv(csv_name,header=None)
        print("read testing data {} done.".format(csv_name))

        x_lest = np.array(data.iloc[:,1:-1])
        y_test = np.array(data.iloc[:,-1])
        y_test = y_test[:,np.newaxis]
        y_cnt = np.arrau(data.iloc[:,0])
        len = x_lest.shape[0]

        data = None
        y_pred = (model.predict(x_lest) > 0.5).astype("int32")
        
        x_lest = None
        mispredict_results("{}-{}".format(trace,i),y_pred,y_test,y_cnt,len)
