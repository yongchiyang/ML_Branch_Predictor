import os
import numpy as np
import pandas as pd
import tensorflow as tf

def mispredict_results(trace,trace_name,y_pred,y_test,y_cnt,len):
    total, mispred = 0,0
    for i in range(len):
        total += y_cnt[i]
        if(y_pred[i] != y_test[i]): mispred += y_cnt[i]
    
    with open("{}-mispredict".format(trace),'a') as file:
        file.write("{0},{1},{2}\n".format(trace_name,total,mispred))

traces = [("LONG_MOBILE",32),("SHORT_SERVER",293),("LONG_SERVER",8),("SHORT_MOBILE",107)]

for trace,iter in traces:
    model = tf.keras.models.load_model('models/{}.keras'.format(trace))
    for i in range(1,iter+1):
        csv_name = os.path.join("../../data/data_info/{}.test/".format(trace),"{}-{}.bt9.trace".format(trace,i))
        data = pd.read_csv(csv_name,header=None)
        print("read testing data {} done.".format(csv_name))

        x_test = np.array(data.iloc[:,1:-1])
        y_test = np.array(data.iloc[:,-1])
        y_test = y_test[:,np.newaxis]
        y_cnt = np.array(data.iloc[:,0])
        len = x_test.shape[0]

        data = None
        y_pred = (model.predict(x_test) > 0.5).astype("int32")
        
        x_test = None
        mispredict_results(trace,"{}-{}".format(trace,i),y_pred,y_test,y_cnt,len)
